"""Train the point-task random forest on the feature table."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click

from complexity.train import run


@click.command()
@click.option("--features", "features_path", type=click.Path(path_type=Path),
              default=Path("data/features/all.parquet"))
@click.option("--config", type=click.Path(path_type=Path),
              default=Path("configs/point_rf.yaml"))
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=Path("models"))
def main(features_path: Path, config: Path, out_dir: Path) -> None:
    run(features_path, config, out_dir)


if __name__ == "__main__":
    main()
