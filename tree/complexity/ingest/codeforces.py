"""Best-effort Codeforces ingest via `open-r1/codeforces` on HuggingFace.

The dataset has problem metadata including `editorial` (markdown text from
official editorials) for ~10k problems, plus solutions. We mine the editorial
text for complexity annotations using the same regex as the doocs/leetcode
miner, then match each annotated problem to its Python solutions via
`open-r1/codeforces-submissions` (filtered to Python submissions).

Yield is modest because only a subset of editorials state complexity in a
grep-friendly form. This is acceptable: Codeforces is a supplementary source
on top of CodeComplex (which already contains Codeforces-origin problems) and
doocs/leetcode.

Labels are origin="comment" since they derive from editorial prose.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..normalizer import normalize
from ..schemas import PointRecord
from .utils import find_complexity_in_text, looks_like_python, sha256_str, write_points, write_rejects


def _ingest_problems(hf_problems: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset(hf_problems, split="train")
    # problem_id -> (label, raw_complexity)
    complexity_by_problem: dict[str, tuple[str, str]] = {}
    rejects: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        problem_id = row.get("id") or row.get("problem_id") or row.get("contest_id")
        if problem_id is None:
            continue
        editorial = row.get("editorial") or row.get("editorial_text") or ""
        notes = row.get("note") or ""
        description = row.get("description") or ""
        for text in (editorial, notes, description):
            if not text:
                continue
            raw = find_complexity_in_text(text)
            if raw is None:
                continue
            label = normalize(raw)
            if label is None:
                rejects.append({"problem_id": str(problem_id),
                                "reason": "normalize_fail", "raw": raw})
                continue
            complexity_by_problem[str(problem_id)] = (label, raw)
            break
    return complexity_by_problem, rejects


def _ingest_submissions(hf_submissions: str,
                         complexity_by_problem: dict[str, tuple[str, str]],
                         limit: int | None):
    from datasets import load_dataset
    ds = load_dataset(hf_submissions, split="train", streaming=True)
    records: list[PointRecord] = []
    rejects: list[dict] = []
    solution_counter: dict[str, int] = {}

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        lang = (row.get("programmingLanguage") or row.get("language") or "").lower()
        if "python" not in lang:
            continue
        problem_id = str(row.get("problem_id") or row.get("problemId")
                          or f"{row.get('contestId')}-{row.get('index')}")
        if problem_id not in complexity_by_problem:
            continue
        code = row.get("source") or row.get("code") or ""
        if not looks_like_python(code):
            continue

        label, raw = complexity_by_problem[problem_id]
        sha = sha256_str(code)
        solution_counter[problem_id] = solution_counter.get(problem_id, 0) + 1
        idx = solution_counter[problem_id] - 1
        records.append(PointRecord(
            id=f"codeforces::{problem_id}::{idx}::{sha[:8]}",
            source="codeforces",
            problem_id=problem_id,
            solution_idx=idx,
            code=code,
            code_sha256=sha,
            label=label,
            raw_complexity=raw,
            origin="comment",
            ast_nodes=0,
        ))

    return records, rejects


@click.command()
@click.option("--hf-problems", default="open-r1/codeforces", show_default=True)
@click.option("--hf-submissions", default="open-r1/codeforces-submissions", show_default=True)
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("data/interim/codeforces.parquet"))
@click.option("--audit-dir", type=click.Path(path_type=Path),
              default=Path("data/audit"))
@click.option("--problems-limit", type=int, default=None,
              help="Cap number of problems scanned (for smoke tests).")
@click.option("--submissions-limit", type=int, default=200_000,
              help="Cap number of submissions scanned.")
@click.option("--per-problem-cap", type=int, default=5,
              help="Max solutions kept per problem (to control dataset balance).")
def main(hf_problems: str, hf_submissions: str, out: Path, audit_dir: Path,
         problems_limit: int | None, submissions_limit: int | None,
         per_problem_cap: int) -> None:
    all_rejects: list[dict] = []

    click.echo(f"[codeforces] scanning problems from {hf_problems}...")
    try:
        complexity_by_problem, pr_rej = _ingest_problems(hf_problems, problems_limit)
        all_rejects.extend(pr_rej)
    except Exception as e:
        click.echo(f"[codeforces] problems ingest failed: {e}")
        complexity_by_problem = {}
        all_rejects.append({"stage": "problems", "reason": "exception", "error": str(e)})

    click.echo(f"[codeforces] {len(complexity_by_problem)} problems with mineable complexity")

    if not complexity_by_problem:
        write_points([], Path(out))
        write_rejects(all_rejects, Path(audit_dir) / "codeforces_rejects.jsonl")
        click.echo("[codeforces] no complexity-labeled problems; skipping submission pass")
        return

    click.echo(f"[codeforces] scanning submissions from {hf_submissions} (cap={submissions_limit})...")
    try:
        records, sub_rej = _ingest_submissions(hf_submissions, complexity_by_problem, submissions_limit)
        all_rejects.extend(sub_rej)
    except Exception as e:
        click.echo(f"[codeforces] submissions ingest failed: {e}")
        records = []
        all_rejects.append({"stage": "submissions", "reason": "exception", "error": str(e)})

    # Per-problem cap to avoid dataset skew from spam-submitted problems.
    if per_problem_cap and records:
        capped: list[PointRecord] = []
        counts: dict[str, int] = {}
        for r in records:
            c = counts.get(r.problem_id or "", 0)
            if c < per_problem_cap:
                capped.append(r)
                counts[r.problem_id or ""] = c + 1
        records = capped

    n = write_points(records, Path(out))
    write_rejects(all_rejects, Path(audit_dir) / "codeforces_rejects.jsonl")
    click.echo(f"[codeforces] wrote {n} records to {out}")


if __name__ == "__main__":
    main()
