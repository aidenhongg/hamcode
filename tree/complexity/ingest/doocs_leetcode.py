"""Mine doocs/leetcode for Python solutions with complexity annotations.

doocs/leetcode stores complexity annotations in the English editorial README
(`README_EN.md`) as LaTeX math: e.g. `Time complexity is $O(n)$`. The actual
Python code lives in sibling `Solution.py` / `Solution2.py` files. So the
miner walks READMEs, extracts the complexity, and pairs it with every Python
solution in the same problem directory.

    solution/0000-0099/0001.Two Sum/
        README_EN.md       <-- complexity lives here
        Solution.py        <-- pair as one PointRecord
        Solution2.py       <-- pair as another PointRecord (same label)

This is the primary source for the 4 multi-variable classes — LeetCode matrix
and two-string problems frequently use explicit `m`/`n` in their English
editorials.
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

_PROBLEM_DIR_RE = re.compile(r"^(\d{4})\.")
_PYTHON_SOLUTION_RE = re.compile(r"^Solution\d*\.py$")


def _ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        click.echo(f"[doocs-leetcode] updating existing clone at {repo_dir}")
        try:
            subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", "main"],
                           check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"[doocs-leetcode] git update failed; using existing clone: {e}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"[doocs-leetcode] shallow-cloning to {repo_dir} (this may take a minute)...")
    subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, str(repo_dir)], check=True)


def _extract_problem_id(path: Path) -> str | None:
    for part in path.parts:
        m = _PROBLEM_DIR_RE.match(part)
        if m:
            return m.group(1)
    return None


def _iter_problem_dirs(repo_dir: Path):
    """Yield each problem directory that contains a README_EN.md."""
    for root in ("solution", "lcof", "lcof2", "lcci", "lcp"):
        base = repo_dir / root
        if not base.exists():
            continue
        for readme in base.rglob("README_EN.md"):
            yield readme.parent
    # Also try the Chinese README as fallback if EN missing in that dir
    for root in ("solution",):
        base = repo_dir / root
        if not base.exists():
            continue
        for readme in base.rglob("README.md"):
            if (readme.parent / "README_EN.md").exists():
                continue  # already processed via EN
            yield readme.parent


def _read_readme(problem_dir: Path) -> str | None:
    for name in ("README_EN.md", "README.md"):
        p = problem_dir / name
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return None


def _python_solutions(problem_dir: Path):
    for p in sorted(problem_dir.iterdir()):
        if p.is_file() and _PYTHON_SOLUTION_RE.match(p.name):
            yield p


@click.command()
@click.option("--repo-dir", type=click.Path(path_type=Path),
              default=Path("data/raw/doocs-leetcode"))
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("data/interim/doocs_leetcode.parquet"))
@click.option("--audit-dir", type=click.Path(path_type=Path),
              default=Path("data/audit"))
@click.option("--max-problems", type=int, default=None,
              help="Cap number of problems scanned (for smoke tests).")
@click.option("--no-clone", is_flag=True,
              help="Skip git clone/update; use existing repo-dir as-is.")
def main(repo_dir: Path, out: Path, audit_dir: Path,
         max_problems: int | None, no_clone: bool) -> None:
    if not no_clone:
        _ensure_repo(repo_dir)
    else:
        if not repo_dir.exists():
            raise click.ClickException(
                f"--no-clone set but repo dir {repo_dir} does not exist"
            )

    records: list[PointRecord] = []
    rejects: list[dict] = []

    problems_scanned = 0
    problems_with_label = 0
    for problem_dir in _iter_problem_dirs(repo_dir):
        if max_problems is not None and problems_scanned >= max_problems:
            break
        problems_scanned += 1

        readme = _read_readme(problem_dir)
        if readme is None:
            rejects.append({"dir": str(problem_dir), "reason": "no_readme"})
            continue

        raw = find_complexity_in_text(readme, max_chars=20000)
        if raw is None:
            rejects.append({"dir": str(problem_dir), "reason": "no_complexity_in_readme"})
            continue

        label = normalize(raw)
        if label is None:
            rejects.append({"dir": str(problem_dir), "reason": "normalize_fail", "raw": raw})
            continue

        pid = _extract_problem_id(problem_dir) or problem_dir.name
        py_files = list(_python_solutions(problem_dir))
        if not py_files:
            rejects.append({"dir": str(problem_dir), "reason": "no_python_solution"})
            continue

        problems_with_label += 1
        for idx, py_file in enumerate(py_files):
            try:
                code = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                rejects.append({"file": str(py_file), "reason": "read_error", "error": str(e)})
                continue
            if not code.strip():
                rejects.append({"file": str(py_file), "reason": "empty"})
                continue
            sha = sha256_str(code)
            records.append(PointRecord(
                id=f"leetcode::{pid}::{idx}::{sha[:8]}",
                source="leetcode",
                problem_id=pid,
                solution_idx=idx,
                code=code,
                code_sha256=sha,
                label=label,
                raw_complexity=raw,
                origin="comment",
                ast_nodes=0,
            ))

    n = write_points(records, Path(out))
    write_rejects(rejects, Path(audit_dir) / "doocs_leetcode_rejects.jsonl")
    click.echo(f"[doocs-leetcode] scanned {problems_scanned} problems, "
               f"{problems_with_label} with mineable complexity + Python solution")
    click.echo(f"[doocs-leetcode] accepted {n} solution records, rejected {len(rejects)}")
    click.echo(f"[doocs-leetcode] wrote {n} records to {out}")


if __name__ == "__main__":
    main()
