"""Ingest CodeComplex from HuggingFace (`codeparrot/codecomplex`).

The HF dataset is currently the Java-only 2022 version (4,200 examples). We
filter to Python defensively: if any row has a `language` field we use it;
otherwise we use the `looks_like_python` heuristic. In practice this ingest
will yield ~0 Python rows from the public HF dataset today, and we rely on
doocs/leetcode + codeforces for Python coverage.

If the user drops an extended CodeComplex Python dump at
`data/raw/codecomplex_python/*.jsonl` with fields {src, complexity, problem},
we also pick those up (future-proofing for when the paper's Python split
becomes publicly available).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from ..normalizer import normalize_any
from ..schemas import PointRecord
from .utils import looks_like_python, sha256_str, write_points, write_rejects


_DEFAULT_HF_NAME = "codeparrot/codecomplex"


def _ingest_from_hf(name: str, limit: int | None) -> tuple[list[PointRecord], list[dict]]:
    from datasets import load_dataset  # lazy import so repo can be browsed offline
    ds = load_dataset(name, split="train")
    records: list[PointRecord] = []
    rejects: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        code = row.get("src") or row.get("code") or ""
        raw = row.get("complexity") or ""
        lang = (row.get("language") or "").lower()
        problem = row.get("problem") or row.get("problem_id") or None
        source_origin = (row.get("from") or "codecomplex").lower()

        if lang and "python" not in lang:
            rejects.append({"idx": i, "reason": "not_python", "lang": lang})
            continue
        if not lang and not looks_like_python(code):
            rejects.append({"idx": i, "reason": "heuristic_not_python"})
            continue

        label = normalize_any(raw)
        if label is None:
            rejects.append({"idx": i, "reason": "normalize_fail", "raw": raw})
            continue

        sha = sha256_str(code)
        rec_id = f"codecomplex::{sha[:12]}"
        records.append(PointRecord(
            id=rec_id,
            source=f"codecomplex/{source_origin}" if source_origin != "codecomplex" else "codecomplex",
            problem_id=str(problem) if problem else None,
            solution_idx=None,
            code=code,
            code_sha256=sha,
            label=label,
            raw_complexity=raw,
            origin="dataset",
            ast_nodes=0,  # populated later in feature-extraction stage
        ))
    return records, rejects


def _ingest_from_local(local_dir: Path) -> tuple[list[PointRecord], list[dict]]:
    records: list[PointRecord] = []
    rejects: list[dict] = []
    if not local_dir.exists():
        return records, rejects
    for jsonl in sorted(local_dir.glob("*.jsonl")):
        with jsonl.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rejects.append({"file": jsonl.name, "line": line_no,
                                    "reason": "json_decode"})
                    continue
                code = row.get("src") or row.get("code") or ""
                raw = row.get("complexity") or ""
                lang = (row.get("language") or "python").lower()
                if "python" not in lang:
                    rejects.append({"file": jsonl.name, "line": line_no,
                                    "reason": "not_python"})
                    continue
                label = normalize_any(raw)
                if label is None:
                    rejects.append({"file": jsonl.name, "line": line_no,
                                    "reason": "normalize_fail", "raw": raw})
                    continue
                sha = sha256_str(code)
                records.append(PointRecord(
                    id=f"codecomplex-local::{sha[:12]}",
                    source="codecomplex",
                    problem_id=str(row.get("problem") or row.get("problem_id") or ""),
                    solution_idx=row.get("solution_idx"),
                    code=code,
                    code_sha256=sha,
                    label=label,
                    raw_complexity=raw,
                    origin="dataset",
                    ast_nodes=0,
                ))
    return records, rejects


@click.command()
@click.option("--hf-name", default=_DEFAULT_HF_NAME, show_default=True,
              help="HuggingFace dataset repo id")
@click.option("--local-dir", type=click.Path(path_type=Path),
              default=Path("data/raw/codecomplex_python"),
              help="Optional local JSONL directory to supplement HF data.")
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("data/interim/codecomplex.parquet"))
@click.option("--audit-dir", type=click.Path(path_type=Path),
              default=Path("data/audit"))
@click.option("--limit", type=int, default=None,
              help="Max rows to ingest from HF (for smoke tests).")
@click.option("--offline", is_flag=True,
              help="Skip HuggingFace; use local_dir only.")
def main(hf_name: str, local_dir: Path, out: Path, audit_dir: Path,
         limit: int | None, offline: bool) -> None:
    records: list[PointRecord] = []
    rejects: list[dict] = []

    if not offline:
        try:
            hf_recs, hf_rej = _ingest_from_hf(hf_name, limit)
            records.extend(hf_recs)
            rejects.extend(hf_rej)
            click.echo(f"[codecomplex] HF: {len(hf_recs)} accepted, {len(hf_rej)} rejected")
        except Exception as e:
            click.echo(f"[codecomplex] HF ingest failed: {e}")
            rejects.append({"reason": "hf_exception", "error": str(e)})

    local_recs, local_rej = _ingest_from_local(Path(local_dir))
    records.extend(local_recs)
    rejects.extend(local_rej)
    if local_recs or local_rej:
        click.echo(f"[codecomplex] local: {len(local_recs)} accepted, {len(local_rej)} rejected")

    n = write_points(records, Path(out))
    write_rejects(rejects, Path(audit_dir) / "codecomplex_rejects.jsonl")
    click.echo(f"[codecomplex] wrote {n} records to {out}")


if __name__ == "__main__":
    main()
