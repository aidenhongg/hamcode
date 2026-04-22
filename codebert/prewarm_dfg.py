"""Pre-compute and cache DFG for every sample in a parquet file.

Massively reduces wallclock on the first epoch: tree-sitter + DFG extraction is
CPU-bound and single-threaded per sample, so running it serially via the
DataLoader starves the GPU. Running it upfront in a multiprocessing pool is
10-20x faster and the cache is shared with training (keyed by code SHA256).

Example:
    python prewarm_dfg.py --parquet data/processed/train.parquet --workers 16
    python prewarm_dfg.py --parquet data/processed/val.parquet --workers 16
    python prewarm_dfg.py --parquet data/processed/test.parquet --workers 16
    python prewarm_dfg.py --parquet data/processed/pair_train.parquet --pair --workers 16

After this, train.py's first epoch runs at full GPU utilization.
"""

from __future__ import annotations

# IMPORTANT: these env vars MUST be set before numpy / scipy / sklearn / torch
# / transformers import, otherwise each worker process grabs a full BLAS
# thread pool and the system runs out of process slots. On a 128-core pod
# with 16 workers, that's 128*16 = 2048 threads contending for CPU.
import os
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "TOKENIZERS_PARALLELISM"):
    os.environ.setdefault(_k, "1" if "THREADS" in _k else "false")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq
from tqdm.auto import tqdm

# Eagerly import heavy deps in the PARENT. With start_method=fork (Linux default),
# child workers inherit these imports via copy-on-write — no re-import cost,
# no second BLAS pool per worker.
def _eager_parent_imports(model_name: str) -> None:
    from transformers import AutoTokenizer  # noqa: F401
    AutoTokenizer.from_pretrained(model_name)   # populate HF cache once
    from data import get_python_parser
    get_python_parser()


def _worker_init(model_name: str, max_seq_len: int, max_dfg_nodes: int, task: str) -> None:
    global _tokenizer, _cfg
    # Re-assert in case spawn method was used (children don't inherit env).
    for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        os.environ[_k] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    from data import get_python_parser
    get_python_parser()
    _cfg = {"max_seq_len": max_seq_len, "max_dfg_nodes": max_dfg_nodes, "task": task}


def _worker_point(code: str) -> bool:
    from common.dfg_cache import DFGCache
    from data import build_point_inputs
    cache = DFGCache()
    key = f"point:{_cfg['max_seq_len']}:{_cfg['max_dfg_nodes']}"
    composite = code + "|" + key
    if composite in cache:
        return False
    bundle = build_point_inputs(code, _tokenizer, _cfg["max_seq_len"], _cfg["max_dfg_nodes"])
    try:
        cache.put(composite, bundle)
    except OSError:
        return False
    return True


def _worker_pair(payload: tuple[str, str]) -> bool:
    from common.dfg_cache import DFGCache
    from data import build_pair_inputs
    code_a, code_b = payload
    cache = DFGCache()
    key = f"pair:{_cfg['max_seq_len']}:{_cfg['max_dfg_nodes']}"
    composite = code_a + "\n---\n" + code_b + "|" + key
    if composite in cache:
        return False
    bundle = build_pair_inputs(
        code_a, code_b, _tokenizer, _cfg["max_seq_len"], _cfg["max_dfg_nodes"]
    )
    try:
        cache.put(composite, bundle)
    except OSError:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--pair", action="store_true",
                    help="Parquet has code_a/code_b (pairwise); default assumes pointwise")
    # Cap at 16 — more processes rarely help (each pays ~500MB RSS for transformers
    # + tokenizer) and on 64+ core boxes they contend on the DFG disk cache.
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 4))))
    ap.add_argument("--chunksize", type=int, default=4,
                    help="smaller = more frequent progress updates (default 4)")
    ap.add_argument("--start_method", default="fork",
                    choices=("fork", "spawn", "forkserver"),
                    help="Linux 'fork' is much faster than 'spawn' for heavy imports")
    ap.add_argument("--serial", action="store_true",
                    help="single-process mode (diagnostic — skips mp.Pool entirely)")
    ap.add_argument("--model_name", default="microsoft/graphcodebert-base")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--max_dfg_nodes", type=int, default=64)
    args = ap.parse_args()

    path = Path(args.parquet)
    table = pq.read_table(path)
    if args.pair:
        payloads = list(zip(table.column("code_a").to_pylist(),
                            table.column("code_b").to_pylist()))
        worker = _worker_pair
        task = "pair"
    else:
        payloads = table.column("code").to_pylist()
        worker = _worker_point
        task = "point"

    print(f"[prewarm] parquet={path}  task={task}  samples={len(payloads)}  "
          f"workers={args.workers}  chunksize={args.chunksize}  "
          f"start_method={args.start_method}  serial={args.serial}", flush=True)
    t0 = time.time()
    n_built = 0
    n_cached = 0

    if args.serial or args.workers <= 1:
        # Diagnostic / fallback path — process in the main process.
        _worker_init(args.model_name, args.max_seq_len, args.max_dfg_nodes, task)
        for item in tqdm(payloads, total=len(payloads), dynamic_ncols=True):
            built = worker(item)
            if built: n_built += 1
            else:      n_cached += 1
    else:
        # Eagerly load heavy modules in parent so fork() children inherit them.
        if args.start_method == "fork":
            print("[prewarm] loading transformers + tree-sitter in parent (fork inherit)...",
                  flush=True)
            _eager_parent_imports(args.model_name)
        ctx = mp.get_context(args.start_method)
        print(f"[prewarm] spawning {args.workers} workers ({args.start_method})...", flush=True)
        with ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(args.model_name, args.max_seq_len, args.max_dfg_nodes, task),
        ) as pool:
            print("[prewarm] pool ready, processing", flush=True)
            for built in tqdm(
                pool.imap_unordered(worker, payloads, chunksize=args.chunksize),
                total=len(payloads), dynamic_ncols=True,
            ):
                if built: n_built += 1
                else:      n_cached += 1

    dt = time.time() - t0
    rate = len(payloads) / max(0.1, dt)
    print(f"[prewarm] done in {dt:.1f}s  new={n_built}  already_cached={n_cached}  "
          f"rate={rate:.1f}/s", flush=True)
    return 0


if __name__ == "__main__":
    # Linux default is fork; we leave it be. On Windows spawn is required but the
    # user's pod is Linux. `--start_method` can override.
    sys.exit(main())
