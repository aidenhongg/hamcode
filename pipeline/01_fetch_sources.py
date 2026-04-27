"""Fetch raw data sources.

Sources fetched (each independently skippable):
  - doocs/leetcode (shallow clone)               -> data/raw/leetcode/
  - sybaik1/CodeComplex-Data Python jsonl        -> data/raw/codecomplex/python.jsonl
  - sybaik1/CodeComplex-Data Java   jsonl        -> data/raw/codecomplex/java.jsonl
  - kamyu104/LeetCode-Solutions (shallow clone)  -> data/raw/kamyu/
  - mxeval / MBXP per-language jsonl             -> data/raw/mbxp/{mbpp,mbjp,mbcpp,...}.jsonl

If your network blocks GitHub or huggingface, you can pre-stage the layout
manually and pass the corresponding --skip_* flags.
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
KAMYU_REPO    = "https://github.com/kamyu104/LeetCode-Solutions.git"

# Order matters - first success wins.
CODECOMPLEX_PY_CANDIDATES: tuple[str, ...] = (
    "https://raw.githubusercontent.com/sybaik1/CodeComplex-Data/main/python_data.jsonl",
    "https://huggingface.co/datasets/codeparrot/codecomplex/resolve/main/data.jsonl",
)
CODECOMPLEX_JAVA_CANDIDATES: tuple[str, ...] = (
    "https://raw.githubusercontent.com/sybaik1/CodeComplex-Data/main/java_data.jsonl",
)

# MBXP / mxeval per-language splits. The primary source is the amazon-science/mxeval
# repo's `data/` directory. We list HTTP URLs for each language we cover.
MBXP_FILES: dict[str, tuple[str, ...]] = {
    "mbpp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbpp_release_v1.jsonl",
    ),
    "mbjp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbjp_release_v1.2.jsonl",
    ),
    "mbjsp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbjsp_release_v1.2.jsonl",
    ),
    "mbtsp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbtsp_release_v1.2.jsonl",
    ),
    "mbcpp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbcpp_release_v1.jsonl",
    ),
    "mbcsp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbcsp_release_v1.jsonl",
    ),
    "mbgp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbgp_release_v1.jsonl",
    ),
    "mbphp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbphp_release_v1.jsonl",
    ),
    "mbrbp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbrbp_release_v1.jsonl",
    ),
    "mbswp.jsonl": (
        "https://raw.githubusercontent.com/amazon-science/mxeval/main/data/mbxp/mbswp_release_v1.jsonl",
    ),
}


def _shallow_clone(url: str, dest: Path, depth: int, force: bool) -> None:
    if dest.exists() and (dest / ".git").exists():
        if force:
            shutil.rmtree(dest)
        else:
            print(f"[01] {url} already cloned at {dest}", flush=True)
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[01] cloning {url} -> {dest} (depth={depth})", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", str(depth), url, str(dest)],
        check=True,
    )


def _http_fetch(url_candidates: tuple[str, ...], dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[01] present: {dest_path} ({dest_path.stat().st_size} bytes)", flush=True)
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for url in url_candidates:
        print(f"[01] try: {url}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "codebert/0.2"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            dest_path.write_bytes(data)
            print(f"[01] wrote {dest_path} ({len(data)} bytes)", flush=True)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[01]   -> failed: {e}", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_leetcode", action="store_true")
    ap.add_argument("--skip_codecomplex", action="store_true")
    ap.add_argument("--skip_kamyu", action="store_true")
    ap.add_argument("--skip_mbxp", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    if not args.skip_leetcode:
        _shallow_clone(LEETCODE_REPO, raw / "leetcode", depth=args.depth, force=args.force)
    if not args.skip_kamyu:
        _shallow_clone(KAMYU_REPO, raw / "kamyu", depth=args.depth, force=args.force)
    if not args.skip_codecomplex:
        cc = raw / "codecomplex"
        ok_py = _http_fetch(CODECOMPLEX_PY_CANDIDATES, cc / "python.jsonl")
        ok_jv = _http_fetch(CODECOMPLEX_JAVA_CANDIDATES, cc / "java.jsonl")
        if not (ok_py or ok_jv):
            print(f"[01] WARNING: no CodeComplex data fetched. "
                  f"Drop the jsonl(s) at {cc}/{{python,java}}.jsonl manually.",
                  flush=True)
    if not args.skip_mbxp:
        mbxp_dir = raw / "mbxp"
        for fname, urls in MBXP_FILES.items():
            _http_fetch(urls, mbxp_dir / fname)
    print("[01] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
