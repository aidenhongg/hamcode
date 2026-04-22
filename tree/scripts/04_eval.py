"""Re-evaluate a saved model on the test split without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from complexity.labels import POINT_LABELS
from complexity.train import evaluate, load_model, split_frame, to_xy


@click.command()
@click.option("--features", "features_path", type=click.Path(path_type=Path),
              default=Path("data/features/all.parquet"))
@click.option("--model-path", type=click.Path(path_type=Path),
              default=Path("models/point_rf.joblib"))
@click.option("--out-file", type=click.Path(path_type=Path),
              default=Path("models/metrics_reeval.json"))
def main(features_path: Path, model_path: Path, out_file: Path) -> None:
    model = load_model(model_path)
    df = pd.read_parquet(features_path)
    _, val_df, test_df = split_frame(df)

    metrics = {}
    if len(val_df):
        X, y = to_xy(val_df)
        metrics["val"] = evaluate(model, X, y)
    if len(test_df):
        X, y = to_xy(test_df)
        metrics["test"] = evaluate(model, X, y)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    click.echo(f"[04_eval] labels: {list(POINT_LABELS)}")
    for split in ("val", "test"):
        if split in metrics:
            m = metrics[split]
            click.echo(f"[04_eval] {split}: acc={m['accuracy']:.4f} "
                       f"f1_macro={m['f1_macro']:.4f} f1_weighted={m['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
