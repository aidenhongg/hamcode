"""Walk a Phase-A run + Phase-B LoRA bundle and produce a side-by-side
per-language comparison.

Inputs
------
  --fullft_run     runs/multi-fullft-{ts}/   (Phase A; reads test_metrics.json)
  --lora_root      runs/lora-{ts}/           (Phase B; reads {lang}/test_metrics.json)
  --out_dir        defaults to lora_root

Outputs
-------
  <out_dir>/bundle_report.json   structured side-by-side metrics
  <out_dir>/bundle_report.md     human-readable table

Designed to fail loud (FATAL: ...) on any missing artifact so the Runpod
script aborts instead of producing a half-empty bundle.

CLI:
    python scripts/report_lora_bundle.py \\
        --fullft_run runs/multi-fullft-20260425 \\
        --lora_root  runs/lora-20260425
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[1]
sys.path.insert(0, str(REPO))

from common.schemas import LANGUAGES


_HEADLINE_KEYS = ("accuracy", "macro_f1", "within_1_tier_accuracy")


def _load_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _phase_a_per_language(fullft_run: Path) -> dict[str, dict]:
    """Read Phase-A test_metrics.json and return its `per_language` dict.

    Falls back to {} if Phase-A was trained before per-language logging
    landed (and its test_metrics has no per_language field) — the report
    still renders, with `-` for the Phase-A column.
    """
    test_metrics = fullft_run / "test_metrics.json"
    if not test_metrics.exists():
        # Some runs nest as fullft_run/<run-name>/test_metrics.json; scan one level.
        for child in fullft_run.iterdir():
            if child.is_dir() and (child / "test_metrics.json").exists():
                test_metrics = child / "test_metrics.json"
                break
    if not test_metrics.exists():
        raise FileNotFoundError(
            f"no test_metrics.json under {fullft_run} - did Phase A run?")
    data = _load_json(test_metrics)
    return data.get("per_language", {}), data


def _phase_b_per_language(lora_root: Path) -> dict[str, dict]:
    """Read each {lang}/test_metrics.json under lora_root and return
    {lang: <flat headline dict>}.  Languages without a metrics file are
    omitted (caller can warn)."""
    out: dict[str, dict] = {}
    for lang in LANGUAGES:
        p = lora_root / lang / "test_metrics.json"
        if not p.exists():
            continue
        data = _load_json(p)
        # The lora_train evaluate() now emits per_language[lang] when
        # `language=` is passed; prefer that over the top-level numbers in
        # case the slice contains rows of multiple languages (shouldn't, but
        # be defensive).
        plang = data.get("per_language", {})
        slice_metrics = plang.get(lang, data)   # fallback to top-level
        out[lang] = {
            "accuracy": slice_metrics.get("accuracy", 0.0),
            "macro_f1": slice_metrics.get("macro_f1", 0.0),
            "within_1_tier_accuracy": slice_metrics.get(
                "within_1_tier_accuracy", 0.0),
            "n": slice_metrics.get("n",
                                    slice_metrics.get("per_class", {}).get(
                                        "support_total", 0)),
        }
    return out


def _fmt(x: float | int | None, fmt: str = "{:.4f}") -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return fmt.format(x)
    return str(x)


def _render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Hamcode bundle report")
    lines.append("")
    lines.append(f"- fullft_run: `{report['fullft_run']}`")
    lines.append(f"- lora_root: `{report['lora_root']}`")
    lines.append("")
    pa_overall = report.get("phase_a_overall", {})
    if pa_overall:
        lines.append("## Phase A — overall test metrics")
        lines.append("")
        lines.append(f"- accuracy: **{_fmt(pa_overall.get('accuracy'))}**")
        lines.append(f"- macro_f1: **{_fmt(pa_overall.get('macro_f1'))}**")
        lines.append(f"- within_1_tier_accuracy: **{_fmt(pa_overall.get('within_1_tier_accuracy'))}**")
        lines.append("")
    lines.append("## Per-language Phase A vs Phase B")
    lines.append("")
    lines.append("| language | n | A acc | A macroF1 | A w1tier | B acc | B macroF1 | B w1tier | ΔmacroF1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    pa = report["phase_a_per_language"]
    pb = report["phase_b_per_language"]
    for lang in LANGUAGES:
        a = pa.get(lang)
        b = pb.get(lang)
        if a is None and b is None:
            continue
        n = (a.get("n") if a else None) or (b.get("n") if b else None) or 0
        a_acc = a.get("accuracy") if a else None
        a_f1  = a.get("macro_f1") if a else None
        a_w1  = a.get("within_1_tier_accuracy") if a else None
        b_acc = b.get("accuracy") if b else None
        b_f1  = b.get("macro_f1") if b else None
        b_w1  = b.get("within_1_tier_accuracy") if b else None
        delta = None
        if a_f1 is not None and b_f1 is not None:
            delta = b_f1 - a_f1
        lines.append(
            "| {lang} | {n} | {a_acc} | {a_f1} | {a_w1} | {b_acc} | {b_f1} | {b_w1} | {delta} |".format(
                lang=lang, n=n,
                a_acc=_fmt(a_acc), a_f1=_fmt(a_f1), a_w1=_fmt(a_w1),
                b_acc=_fmt(b_acc), b_f1=_fmt(b_f1), b_w1=_fmt(b_w1),
                delta=_fmt(delta, "{:+.4f}") if delta is not None else "-",
            )
        )
    lines.append("")
    if report.get("warnings"):
        lines.append("## Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fullft_run", required=True,
                    help="Phase-A run dir (contains test_metrics.json)")
    ap.add_argument("--lora_root", required=True,
                    help="Phase-B bundle root (contains {lang}/test_metrics.json)")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write bundle_report.{json,md}; default = lora_root")
    args = ap.parse_args()

    fullft_run = Path(args.fullft_run)
    lora_root = Path(args.lora_root)
    out_dir = Path(args.out_dir) if args.out_dir else lora_root
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    try:
        pa_per_lang, pa_full = _phase_a_per_language(fullft_run)
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1
    if not pa_per_lang:
        warnings.append(
            "Phase-A test_metrics.json has no `per_language` field — older "
            "checkpoint? per-language A column will show '-'."
        )

    pb_per_lang = _phase_b_per_language(lora_root)
    missing = [l for l in LANGUAGES if l not in pb_per_lang]
    if missing:
        warnings.append(
            f"Phase-B missing test_metrics.json for: {missing}. "
            f"Did `lora_train.py` finish for those languages?"
        )

    pa_overall = {k: pa_full.get(k) for k in _HEADLINE_KEYS}

    report = {
        "fullft_run": str(fullft_run),
        "lora_root": str(lora_root),
        "phase_a_overall": pa_overall,
        "phase_a_per_language": pa_per_lang,
        "phase_b_per_language": pb_per_lang,
        "warnings": warnings,
    }

    (out_dir / "bundle_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "bundle_report.md").write_text(
        _render_md(report), encoding="utf-8")

    # Pretty-print to stdout too (visible in run_runpod.sh logs).
    print(_render_md(report))
    return 0 if not warnings else 0   # warnings are non-fatal


if __name__ == "__main__":
    sys.exit(main())
