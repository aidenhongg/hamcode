"""Run all three ingest sub-pipelines and merge their outputs.

This is a thin orchestrator — each sub-pipeline can be run independently via
its own module (`python -m complexity.ingest.codecomplex`, etc.). This script
is the one-shot "pull everything" option.

Output:
    data/interim/all.parquet — merged PointRecords from all sources, deduped
                                by code_sha256.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import pandas as pd


INTERIM = Path("data/interim")
SOURCES = [
    ("codecomplex", "complexity.ingest.codecomplex"),
    ("doocs_leetcode", "complexity.ingest.doocs_leetcode"),
    ("codeforces", "complexity.ingest.codeforces"),
]


def _run_module(module: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *extra_args]
    click.echo(f"[01_ingest] running: {' '.join(cmd)}")
    return subprocess.call(cmd)


@click.command()
@click.option("--skip", multiple=True,
              type=click.Choice([name for name, _ in SOURCES]),
              help="Skip a source. Repeatable.")
@click.option("--codecomplex-offline", is_flag=True,
              help="Pass --offline to the codecomplex ingest (use local data only).")
@click.option("--codeforces-submissions-limit", type=int, default=200_000)
@click.option("--doocs-max-files", type=int, default=None)
@click.option("--out", type=click.Path(path_type=Path),
              default=INTERIM / "all.parquet")
def main(skip: tuple[str, ...], codecomplex_offline: bool,
         codeforces_submissions_limit: int, doocs_max_files: int | None,
         out: Path) -> None:

    for name, module in SOURCES:
        if name in skip:
            click.echo(f"[01_ingest] skipping {name}")
            continue
        args: list[str] = []
        if name == "codecomplex" and codecomplex_offline:
            args.append("--offline")
        if name == "codeforces":
            args.extend(["--submissions-limit", str(codeforces_submissions_limit)])
        if name == "doocs_leetcode" and doocs_max_files is not None:
            args.extend(["--max-files", str(doocs_max_files)])
        rc = _run_module(module, args)
        if rc != 0:
            click.echo(f"[01_ingest] {name} exited with code {rc} — continuing")

    frames = []
    for name, _ in SOURCES:
        p = INTERIM / f"{name}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
            click.echo(f"[01_ingest] loaded {len(frames[-1])} rows from {p}")
        else:
            click.echo(f"[01_ingest] {p} does not exist — source produced no records")

    if not frames:
        click.echo("[01_ingest] nothing to merge; exiting")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["code_sha256"], keep="first").reset_index(drop=True)
    click.echo(f"[01_ingest] dedup: {before} -> {len(merged)} rows")

    # Per-label summary
    click.echo("[01_ingest] per-label counts:")
    for label, cnt in merged["label"].value_counts().items():
        click.echo(f"           {label:22s} {cnt}")

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    click.echo(f"[01_ingest] wrote merged {len(merged)} rows to {out}")


if __name__ == "__main__":
    main()
