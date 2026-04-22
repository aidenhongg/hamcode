"""Fetch raw data sources.

Clones doocs/leetcode (shallow) into data/raw/leetcode.
Fetches CodeComplex Python jsonl if a working URL is reachable; otherwise
prints placement instructions and continues.

If your network blocks GitHub, clone manually and pass --skip_leetcode.
If CodeComplex auto-fetch fails, drop the jsonl at
    data/raw/codecomplex/python.jsonl
and re-run with --skip_codecomplex.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

LEETCODE_REPO = "https://github.com/doocs/leetcode.git"

# Order matters — first success wins. Extend as more mirrors appear.
CODECOMPLEX_CANDIDATES: tuple[str, ...] = (
    "https://raw.githubusercontent.com/yonsei-toc/CodeComplex/main/dataset/python.jsonl",
    "https://huggingface.co/datasets/code-complex/codecomplex/resolve/main/python.jsonl",
)


def clone_leetcode(dest: Path, depth: int, force: bool) -> None:
    if dest.exists() and (dest / ".git").exists():
        if force:
            shutil.rmtree(dest)
        else:
            print(f"[01] doocs/leetcode already cloned at {dest}", flush=True)
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[01] cloning {LEETCODE_REPO} -> {dest} (depth={depth})", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", str(depth), LEETCODE_REPO, str(dest)],
        check=True,
    )


def fetch_codecomplex(dest_path: Path) -> None:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[01] codecomplex present: {dest_path} ({dest_path.stat().st_size} bytes)", flush=True)
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for url in CODECOMPLEX_CANDIDATES:
        print(f"[01] try codecomplex: {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "codebert/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            dest_path.write_bytes(data)
            print(f"[01] wrote {dest_path} ({len(data)} bytes)", flush=True)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[01]   -> failed: {e}", flush=True)
    print(
        f"[01] WARNING: could not fetch codecomplex. "
        f"Place it manually at {dest_path} and rerun with --skip_codecomplex.",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--skip_leetcode", action="store_true")
    ap.add_argument("--skip_codecomplex", action="store_true")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    if not args.skip_leetcode:
        clone_leetcode(raw / "leetcode", depth=args.depth, force=args.force)
    if not args.skip_codecomplex:
        fetch_codecomplex(raw / "codecomplex" / "python.jsonl")
    print("[01] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
