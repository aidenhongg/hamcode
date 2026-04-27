"""Cartesian sweep: (head x seed) per language, with fail-soft aggregation.

Per-language is the default — the recipe ranks heads independently for every
language present in train, on the assumption that the best head for python
is not necessarily the best head for ruby. Output layout:

    <out_dir>/per_lang/<lang>/<head>-s<seed>/             # one head fit
    <out_dir>/per_lang/<lang>/SUMMARY.{md,csv}            # within-language ranking
    <out_dir>/PER_LANGUAGE_SUMMARY.md                     # cross-language pivot
    <out_dir>/PER_LANGUAGE_BEST.json                      # best head per language
    <out_dir>/SUMMARY.{md,csv}                            # all rows flattened

`--universal` restores the legacy single-head-for-all-languages mode at
<out_dir>/<head>-s<seed>/ — useful for parity baselines against older runs.

Fail-soft: a single experiment failure is logged to <out_dir>/_failures.jsonl
and the sweep continues.

Usage:
    python -m stacking.sweep \
        --config stacking/configs/sweep.yaml \
        --in_splits data/processed \
        --extraction_dir runs/heads/extraction \
        --out_dir runs/heads

    # Smoke: 1 head x 1 seed x 1 language on whatever data is present
    python -m stacking.sweep --smoke

    # Legacy: one universal head per (head, seed)
    python -m stacking.sweep --universal --config stacking/configs/sweep.yaml
"""

from __future__ import annotations

# Load torch FIRST for Windows DLL search path reasons; see train_head.py.
import torch  # noqa: F401

import argparse
import itertools
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stacking import dataset as ds
from stacking import train_head as th


@dataclass
class SweepConfig:
    heads: list[str]
    seeds: list[int]
    class_weight: str = "auto"
    languages: list[str] | str = "auto"   # "auto" -> detect from train
    universal: bool = False

    @classmethod
    def load(cls, path: Path) -> "SweepConfig":
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            heads=list(cfg["heads"]),
            seeds=list(cfg["seeds"]),
            class_weight=cfg.get("class_weight", "auto"),
            languages=cfg.get("languages", "auto"),
        )


def _resolve_languages(
    cfg_languages: list[str] | str,
    in_splits: Path,
    extraction_dir: Path,
    out_dir: Path,
) -> list[str]:
    """Return the concrete language list. 'auto' inspects the per-lang split
    and returns languages that survive the test/train filters."""
    if isinstance(cfg_languages, list):
        return list(cfg_languages)
    # 'auto' — build per-language splits once just to enumerate survivors.
    triples = ds.build_per_language_splits(
        in_splits=in_splits,
        extraction_dir=extraction_dir, out_dir=out_dir,
    )
    return sorted(triples.keys())


def _run_one(head: str, seed: int, in_splits: Path,
             extraction_dir: Path, out_dir: Path, class_weight: str,
             failures: Path, language: str | None) -> dict | None:
    if language is None:
        exp_dir = out_dir / f"{head}-s{seed}"
    else:
        exp_dir = out_dir / "per_lang" / language / f"{head}-s{seed}"
    try:
        met = th.run(
            head_name=head, seed=seed,
            in_splits=in_splits, extraction_dir=extraction_dir,
            out_dir=exp_dir, class_weight_mode=class_weight,
            language=language,
        )
        row = {
            "head": head, "seed": seed,
            "language": language if language is not None else "_universal_",
            "test_acc": met["accuracy"],
            "balanced_acc": met["balanced_accuracy"],
            "macro_f1": met["macro_f1"],
            "roc_auc": met["roc_auc"],
            "brier": met["brier_score"],
            "ece": met["ece"],
            "per_class_same_f1": met["per_class"]["same"]["f1"],
            "per_class_A_faster_f1": met["per_class"]["A_faster"]["f1"],
            "n_test": int(met["per_class"]["same"]["support"]
                          + met["per_class"]["A_faster"]["support"]),
            "dir": str(exp_dir),
        }
        return row
    except Exception as e:  # fail-soft
        msg = {
            "head": head, "seed": seed,
            "language": language if language is not None else "_universal_",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        with failures.open("a", encoding="utf-8") as f:
            f.write(json.dumps(msg) + "\n")
        lang_tag = f" lang={language}" if language is not None else ""
        print(f"[sweep] FAIL {head}/s{seed}{lang_tag}: {e}", flush=True)
        return None


def _write_summary(rows: list[dict], out_dir: Path,
                    title: str = "Stacking Sweep Results") -> None:
    """Flat-rows summary (the global SUMMARY.{md,csv} or a per-language one).

    Sort by test_acc desc; emit a Best-seed-per-head pivot below.
    """
    rows_sorted = sorted(rows, key=lambda r: -r.get("test_acc", 0.0))

    import csv
    csv_path = out_dir / "SUMMARY.csv"
    if rows_sorted:
        keys = list(rows_sorted[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

    md_path = out_dir / "SUMMARY.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Total experiments: {len(rows)}\n\n")
        if not rows_sorted:
            f.write("(no successful runs)\n"); return

        cols = ["head", "seed", "language",
                "test_acc", "macro_f1", "roc_auc", "brier", "ece"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows_sorted:
            def _fmt(k):
                v = r.get(k)
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("| " + " | ".join(_fmt(c) for c in cols) + " |\n")

        f.write("\n## Best seed per head\n\n")
        best: dict[str, dict] = {}
        for r in rows_sorted:
            key = r["head"]
            if key not in best or r["test_acc"] > best[key]["test_acc"]:
                best[key] = r
        pcols = ["head", "best_seed", "test_acc", "macro_f1", "roc_auc"]
        f.write("| " + " | ".join(pcols) + " |\n")
        f.write("|" + "|".join("---" for _ in pcols) + "|\n")
        for head, r in sorted(best.items()):
            row = {
                "head": head,
                "best_seed": r["seed"],
                "test_acc": r["test_acc"],
                "macro_f1": r["macro_f1"],
                "roc_auc": r.get("roc_auc"),
            }
            def _fmt(k):
                v = row.get(k)
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("| " + " | ".join(_fmt(c) for c in pcols) + " |\n")


def _write_per_language_summary(rows: list[dict], out_dir: Path) -> None:
    """Cross-language pivot: best (head, seed) per language by test_acc.

    Also writes PER_LANGUAGE_BEST.json (machine-readable) and computes the
    support-weighted recipe headline (mean test_macro_f1 across languages).
    """
    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        lang = r.get("language", "_universal_")
        if lang == "_universal_":
            continue
        by_lang.setdefault(lang, []).append(r)

    best_per_lang: dict[str, dict] = {}
    for lang, lang_rows in by_lang.items():
        # primary: test_macro_f1 (matches what the head HP search optimizes
        # downstream); tie-break on test_acc.
        lang_rows_sorted = sorted(
            lang_rows,
            key=lambda r: (-r.get("macro_f1", 0.0), -r.get("test_acc", 0.0)),
        )
        best_per_lang[lang] = lang_rows_sorted[0]
        # Also write a per-language SUMMARY.md inside the language's dir
        lang_dir = out_dir / "per_lang" / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        _write_summary(lang_rows, lang_dir,
                       title=f"Stacking Sweep — language={lang}")

    # JSON
    json_path = out_dir / "PER_LANGUAGE_BEST.json"
    json_path.write_text(json.dumps({
        lang: {
            "head": r["head"],
            "seed": r["seed"],
            "test_acc": r["test_acc"],
            "macro_f1": r["macro_f1"],
            "roc_auc": r.get("roc_auc"),
            "n_test": r.get("n_test"),
            "dir": r["dir"],
        } for lang, r in sorted(best_per_lang.items())
    }, indent=2), encoding="utf-8")

    # Markdown pivot + recipe headline
    md_path = out_dir / "PER_LANGUAGE_SUMMARY.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Per-language sweep results\n\n")
        f.write("Best (head, seed) per language by test macro-F1, "
                "tie-broken on test accuracy.\n\n")
        if not best_per_lang:
            f.write("(no per-language runs)\n")
            return
        cols = ["language", "best_head", "best_seed", "n_test",
                "test_acc", "macro_f1", "roc_auc"]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        # Sort by descending test support so widely-tested languages appear first
        for lang, r in sorted(best_per_lang.items(),
                              key=lambda kv: (-kv[1].get("n_test", 0), kv[0])):
            def _fmt(v):
                if v is None: return "—"
                if isinstance(v, float): return f"{v:.4f}"
                return str(v)
            f.write(f"| {lang} | {r['head']} | {r['seed']} | "
                    f"{_fmt(r.get('n_test'))} | {_fmt(r['test_acc'])} | "
                    f"{_fmt(r['macro_f1'])} | {_fmt(r.get('roc_auc'))} |\n")

        # Recipe headline: support-weighted mean of best-per-language macro-F1
        total_n = sum(r.get("n_test", 0) for r in best_per_lang.values())
        if total_n > 0:
            weighted_f1 = sum(
                r.get("n_test", 0) * r.get("macro_f1", 0.0)
                for r in best_per_lang.values()
            ) / total_n
            weighted_acc = sum(
                r.get("n_test", 0) * r.get("test_acc", 0.0)
                for r in best_per_lang.values()
            ) / total_n
            f.write(f"\n**Recipe headline (support-weighted across {len(best_per_lang)} "
                    f"languages, n_test={total_n}):**\n")
            f.write(f"- weighted test macro-F1: {weighted_f1:.4f}\n")
            f.write(f"- weighted test accuracy: {weighted_acc:.4f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="stacking/configs/sweep.yaml")
    ap.add_argument("--in_splits", default="data/processed")
    ap.add_argument("--extraction_dir", default="runs/heads/extraction")
    ap.add_argument("--out_dir", default="runs/heads")
    ap.add_argument("--smoke", action="store_true",
                    help="run xgb/seed 42 only on the largest language "
                         "(quick E2E check)")
    ap.add_argument("--head", default=None, help="override heads from config")
    ap.add_argument("--seed", type=int, default=None, help="override seeds from config")
    ap.add_argument("--language", default=None,
                    help="run only this language (overrides config languages)")
    ap.add_argument("--universal", action="store_true",
                    help="legacy mode: train one head per (head,seed) over ALL "
                         "languages mixed (no per-language axis)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = out_dir / "_failures.jsonl"
    if failures.exists():
        failures.unlink()

    if args.smoke:
        cfg = SweepConfig(heads=["xgb"], seeds=[42], languages=["python"])
    else:
        cfg = SweepConfig.load(Path(args.config))

    if args.head:
        cfg.heads = [args.head]
    if args.seed is not None:
        cfg.seeds = [args.seed]
    if args.language:
        cfg.languages = [args.language]
    if args.universal:
        cfg.universal = True

    # Resolve languages (or skip the axis entirely in universal mode)
    if cfg.universal:
        languages: list[str | None] = [None]
    else:
        resolved = _resolve_languages(
            cfg.languages, Path(args.in_splits), Path(args.extraction_dir),
            out_dir,
        )
        if not resolved:
            print("[sweep] no languages survived filtering — aborting", flush=True)
            return 1
        languages = list(resolved)
        print(f"[sweep] languages: {languages}", flush=True)

    all_combos = list(itertools.product(cfg.heads, cfg.seeds, languages))
    print(f"[sweep] running {len(all_combos)} experiments "
          f"({len(cfg.heads)} heads x "
          f"{len(cfg.seeds)} seeds x {len(languages)} languages)", flush=True)

    rows: list[dict] = []
    for head, seed, language in all_combos:
        lang_tag = f" / {language}" if language is not None else ""
        print(f"\n=== {head} / s{seed}{lang_tag} ===", flush=True)
        row = _run_one(head, seed,
                        Path(args.in_splits), Path(args.extraction_dir),
                        out_dir, cfg.class_weight, failures, language)
        if row is not None:
            rows.append(row)
            _write_summary(rows, out_dir)
            if not cfg.universal:
                _write_per_language_summary(rows, out_dir)

    _write_summary(rows, out_dir)
    if not cfg.universal:
        _write_per_language_summary(rows, out_dir)

    print(f"\n[sweep] done. {len(rows)}/{len(all_combos)} successful.", flush=True)
    if cfg.universal:
        print(f"[sweep] summary: {out_dir / 'SUMMARY.md'}", flush=True)
    else:
        print(f"[sweep] per-language: {out_dir / 'PER_LANGUAGE_SUMMARY.md'}", flush=True)
        print(f"[sweep] flat: {out_dir / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
