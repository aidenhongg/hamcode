"""Extract 18 tree-sitter features for every merged PointRecord and assign splits.

Inputs:
    data/interim/all.parquet
    configs/point_rf.yaml

Outputs:
    data/features/all.parquet   — features + label + split, one row per record
    data/audit/feature_stats.json — distribution + rejection counts

Steps:
    1. Load merged records.
    2. Apply length and AST-node filters from config.
    3. Extract the 18 features per record.
    4. Assign train/val/test split by problem_id (grouped to avoid leakage).
    5. Write the final table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click
import pandas as pd
import yaml
from tqdm import tqdm

from complexity.features import FEATURE_NAMES, extract_features_with_ast_count
from complexity.split import assign_splits, class_distribution


@click.command()
@click.option("--input", "input_path", type=click.Path(path_type=Path),
              default=Path("data/interim/all.parquet"))
@click.option("--config", type=click.Path(path_type=Path),
              default=Path("configs/point_rf.yaml"))
@click.option("--out", type=click.Path(path_type=Path),
              default=Path("data/features/all.parquet"))
@click.option("--audit-dir", type=click.Path(path_type=Path),
              default=Path("data/audit"))
def main(input_path: Path, config: Path, out: Path, audit_dir: Path) -> None:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    max_chars = int(cfg.get("max_code_chars", 20_000))
    min_ast = int(cfg.get("min_ast_nodes", 5))

    click.echo(f"[02_features] loading {input_path}")
    df = pd.read_parquet(input_path)

    rejects: list[dict] = []
    rows: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        code = row["code"]
        if not isinstance(code, str) or not code:
            rejects.append({"id": row.get("id"), "reason": "empty_code"})
            continue
        if len(code) > max_chars:
            rejects.append({"id": row.get("id"), "reason": "too_long",
                            "chars": len(code)})
            continue
        try:
            feats, ast_n = extract_features_with_ast_count(code)
        except Exception as e:
            rejects.append({"id": row.get("id"), "reason": "parse_error",
                            "error": str(e)})
            continue
        if ast_n < min_ast:
            rejects.append({"id": row.get("id"), "reason": "too_few_ast_nodes",
                            "ast_n": ast_n})
            continue
        out_row = row.to_dict()
        out_row.update(feats)
        out_row["ast_nodes"] = ast_n
        rows.append(out_row)

    if not rows:
        raise click.ClickException("no rows survived feature extraction")

    feat_df = pd.DataFrame(rows)

    # Assign splits (grouped by problem_id to avoid leakage)
    feat_df = assign_splits(
        feat_df,
        train_ratio=cfg.get("train_ratio", 0.80),
        val_ratio=cfg.get("val_ratio", 0.10),
        seed=cfg.get("seed", 42),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out, index=False)

    # Audit: per-class count in each split
    dist = class_distribution(feat_df)
    audit_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "total": int(len(feat_df)),
        "rejects": len(rejects),
        "per_source": feat_df["source"].value_counts().to_dict(),
        "per_label": feat_df["label"].value_counts().to_dict(),
        "per_split": feat_df["split"].value_counts().to_dict(),
        "per_label_per_split": dist.to_dict(),
        "feature_names": list(FEATURE_NAMES),
    }
    (audit_dir / "feature_stats.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    if rejects:
        with (audit_dir / "feature_rejects.jsonl").open("w", encoding="utf-8") as f:
            for r in rejects:
                f.write(json.dumps(r, default=str) + "\n")

    click.echo(f"[02_features] wrote {len(feat_df)} rows to {out}")
    click.echo(f"[02_features] rejects: {len(rejects)}")
    click.echo("[02_features] class distribution per split:")
    click.echo(dist.to_string())


if __name__ == "__main__":
    main()
