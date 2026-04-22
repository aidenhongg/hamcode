"""One-off predict: read a Python file or stdin, print predicted class + probs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from complexity.predict import ComplexityPredictor


@click.command()
@click.option("--model-path", type=click.Path(path_type=Path),
              default=Path("models/point_rf.joblib"))
@click.option("--input", "input_path", type=click.Path(path_type=Path),
              default=None, help="Python file to classify. Omit to read stdin.")
@click.option("--json-output", is_flag=True, help="Emit full JSON instead of a summary.")
def main(model_path: Path, input_path: Path | None, json_output: bool) -> None:
    if input_path is not None:
        code = Path(input_path).read_text(encoding="utf-8", errors="replace")
    else:
        code = sys.stdin.read()

    predictor = ComplexityPredictor(model_path)
    result = predictor.predict_one(code)

    if json_output:
        click.echo(json.dumps(result, indent=2))
        return

    click.echo(f"label       : {result['label']}")
    click.echo(f"confidence  : {result['confidence']:.3f}")
    click.echo("top-3       :")
    top = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    for label, prob in top:
        click.echo(f"  {label:22s} {prob:.3f}")


if __name__ == "__main__":
    main()
