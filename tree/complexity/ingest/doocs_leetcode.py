"""Mine doocs/leetcode for Python solutions with complexity annotations.

Strategy:
    1. Clone/update https://github.com/doocs/leetcode (shallow).
    2. Walk `solution/**/*.py` files.
    3. For each file, scan the first ~2KB for a `Time complexity: O(...)` or
       `# Time: O(...)` style annotation.
    4. Normalize the captured O(...) expression to our 11-class label space.
    5. Emit PointRecord with origin="comment".

This is the primary source for the 4 multi-variable classes — LeetCode problems
on matrices, two-string comparisons, and graphs frequently use explicit `m`/`n`
in their editorial complexity annotations.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import click

from ..normalizer import normalize
from ..schemas import PointRecord
from .utils import find_complexity_in_text, sha256_str, write_points, write_rejects


_REPO_URL = "https://github.com/doocs/leetcode.git"

# Match the LeetCode problem directory: `solution/0000-0099/0001.Two Sum/Solution.py`
_PROBLEM_DIR_RE = re.compile(r"(\d{4})\.")


def _ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        click.echo(f"[doocs-leetcode] updating existing clone at {repo_dir}")
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", "main"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"],
                       check=True, capture_output=True)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"[doocs-leetcode] shallow-cloning to {repo_dir} (this may take a minute)...")
    subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, str(repo_dir)],
                   check=True)


def _extract_problem_id(path: Path) -> str | None:
    # Look for the 4-digit problem number in any path segment
    for part in path.parts:
        m = _PROBLEM_DIR_RE.match(part)
        if m:
            return m.group(1)
    return None


def iter_python_solutions(repo_dir: Path, relative_solution_root: str = "solution"):
    root = repo_dir / relative_solution_root
    if not root.exists():
        # Try lcof / lcof2 / lcp too
        for alt in ("solution", "lcof", "lcof2", "lcp", "lcci"):
            ar = repo_dir / alt
            if ar.exists():
                yield from _iter_py(ar)
        return
    yield from _iter_py(root)


def _iter_py(root: Path):
    for p in root.rglob("*.py"):
        # Skip __init__.py and test files
        if p.name.startswith("__") or "test" in p.name.lower():
            continue
        yield p


@click.command()
@click.option("--repo-dir", type=click.Path(path_type=Path),
              default=Path("data/raw/doocs-leetcode"))
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("data/interim/doocs_leetcode.parquet"))
@click.option("--audit-dir", type=click.Path(path_type=Path),
              default=Path("data/audit"))
@click.option("--max-files", type=int, default=None,
              help="Cap number of Python files scanned (for smoke tests).")
@click.option("--no-clone", is_flag=True,
              help="Skip git clone/update; use existing repo-dir as-is.")
def main(repo_dir: Path, out: Path, audit_dir: Path,
         max_files: int | None, no_clone: bool) -> None:
    if not no_clone:
        _ensure_repo(repo_dir)
    else:
        if not repo_dir.exists():
            raise click.ClickException(
                f"--no-clone set but repo dir {repo_dir} does not exist"
            )

    records: list[PointRecord] = []
    rejects: list[dict] = []
    solution_counter: dict[str, int] = {}  # problem_id -> count (for solution_idx)

    seen = 0
    for i, py_file in enumerate(iter_python_solutions(repo_dir)):
        if max_files is not None and seen >= max_files:
            break
        seen += 1
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            rejects.append({"file": str(py_file), "reason": "read_error", "error": str(e)})
            continue

        if not text.strip():
            rejects.append({"file": str(py_file), "reason": "empty"})
            continue

        raw = find_complexity_in_text(text)
        if raw is None:
            rejects.append({"file": str(py_file), "reason": "no_complexity_comment"})
            continue

        label = normalize(raw)
        if label is None:
            rejects.append({"file": str(py_file), "reason": "normalize_fail", "raw": raw})
            continue

        pid = _extract_problem_id(py_file) or "unknown"
        solution_counter[pid] = solution_counter.get(pid, 0) + 1
        sol_idx = solution_counter[pid] - 1

        sha = sha256_str(text)
        rec_id = f"leetcode::{pid}::{sol_idx}::{sha[:8]}"
        records.append(PointRecord(
            id=rec_id,
            source="leetcode",
            problem_id=pid,
            solution_idx=sol_idx,
            code=text,
            code_sha256=sha,
            label=label,
            raw_complexity=raw,
            origin="comment",
            ast_nodes=0,
        ))

    n = write_points(records, Path(out))
    write_rejects(rejects, Path(audit_dir) / "doocs_leetcode_rejects.jsonl")
    click.echo(f"[doocs-leetcode] scanned {seen} files, "
               f"accepted {n}, rejected {len(rejects)}")
    click.echo(f"[doocs-leetcode] wrote {n} records to {out}")


if __name__ == "__main__":
    main()
