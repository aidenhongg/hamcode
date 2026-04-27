"""Smoke tests for stacking.sweep — per-language Cartesian + summary writers."""

from __future__ import annotations

import torch  # noqa: F401

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stacking import sweep


def test_per_language_summary_picks_best_per_lang():
    """_write_per_language_summary picks the best (head, seed) per language
    by macro-F1 and writes PER_LANGUAGE_BEST.json + the markdown pivot."""
    rows = [
        # python: xgb @ s42 wins
        {"head": "xgb", "seed": 42, "language": "python",
         "test_acc": 0.91, "macro_f1": 0.90, "roc_auc": 0.93,
         "n_test": 350, "dir": "/tmp/python/xgb-s42"},
        {"head": "xgb", "seed": 43, "language": "python",
         "test_acc": 0.89, "macro_f1": 0.87, "roc_auc": 0.92,
         "n_test": 350, "dir": "/tmp/python/xgb-s43"},
        {"head": "mlp", "seed": 42, "language": "python",
         "test_acc": 0.85, "macro_f1": 0.83, "roc_auc": 0.88,
         "n_test": 350, "dir": "/tmp/python/mlp-s42"},
        # ruby: mlp @ s44 wins
        {"head": "xgb", "seed": 42, "language": "ruby",
         "test_acc": 0.78, "macro_f1": 0.74, "roc_auc": 0.81,
         "n_test": 71, "dir": "/tmp/ruby/xgb-s42"},
        {"head": "mlp", "seed": 44, "language": "ruby",
         "test_acc": 0.82, "macro_f1": 0.80, "roc_auc": 0.85,
         "n_test": 71, "dir": "/tmp/ruby/mlp-s44"},
    ]
    td = tempfile.mkdtemp()
    try:
        out = Path(td)
        sweep._write_per_language_summary(rows, out)

        # JSON best-per-lang
        best = json.loads((out / "PER_LANGUAGE_BEST.json").read_text(encoding="utf-8"))
        assert best["python"]["head"] == "xgb"
        assert best["python"]["seed"] == 42
        assert best["ruby"]["head"] == "mlp"
        assert best["ruby"]["seed"] == 44

        # Per-language SUMMARY.md exists for each language
        assert (out / "per_lang" / "python" / "SUMMARY.md").exists()
        assert (out / "per_lang" / "ruby" / "SUMMARY.md").exists()

        # Cross-language pivot mentions both + recipe headline
        body = (out / "PER_LANGUAGE_SUMMARY.md").read_text(encoding="utf-8")
        assert "python" in body and "ruby" in body
        assert "Recipe headline" in body
        # Support-weighted: (350*0.90 + 71*0.80) / (350+71) ~= 0.883
        assert "0.883" in body or "0.8831" in body
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_per_language_summary_skips_universal_rows():
    """Rows tagged as language=_universal_ shouldn't pollute the per-language pivot."""
    rows = [
        {"head": "xgb", "seed": 42, "language": "python",
         "test_acc": 0.91, "macro_f1": 0.90, "roc_auc": 0.93,
         "n_test": 100, "dir": "/tmp/python/xgb"},
        {"head": "xgb", "seed": 42, "language": "_universal_",
         "test_acc": 0.85, "macro_f1": 0.84, "roc_auc": 0.89,
         "n_test": 1000, "dir": "/tmp/xgb"},
    ]
    td = tempfile.mkdtemp()
    try:
        out = Path(td)
        sweep._write_per_language_summary(rows, out)
        best = json.loads((out / "PER_LANGUAGE_BEST.json").read_text(encoding="utf-8"))
        assert "_universal_" not in best
        assert "python" in best
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_sweep_config_loads_languages():
    """SweepConfig.load picks up the languages key (default 'auto')."""
    import yaml
    td = tempfile.mkdtemp()
    try:
        cfg_path = Path(td) / "sweep.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "heads": ["xgb"], "seeds": [42],
            "class_weight": "auto", "languages": ["python", "java"],
        }), encoding="utf-8")
        cfg = sweep.SweepConfig.load(cfg_path)
        assert cfg.languages == ["python", "java"]

        # Default 'auto' when key missing
        cfg_path.write_text(yaml.safe_dump({
            "heads": ["xgb"], "seeds": [42],
            "class_weight": "auto",
        }), encoding="utf-8")
        cfg = sweep.SweepConfig.load(cfg_path)
        assert cfg.languages == "auto"
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_sweep_config_has_no_variants_field():
    """The variants axis was retired; loading a config that includes it
    should still work (extra keys ignored), but SweepConfig has no
    `variants` attribute."""
    cfg = sweep.SweepConfig(heads=["xgb"], seeds=[42])
    assert not hasattr(cfg, "variants")


def test_sweep_module_no_longer_references_variant_axis():
    """The retired Variant Literal must be gone from stacking.dataset."""
    from stacking import dataset as ds
    assert not hasattr(ds, "Variant")
    assert not hasattr(ds, "VARIANT_DESCRIPTIONS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
