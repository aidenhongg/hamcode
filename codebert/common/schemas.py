"""Dataset record schemas — dataclasses plus matching pyarrow schemas for parquet IO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pyarrow as pa


# Canonical language identifiers used everywhere downstream.
LANGUAGES: tuple[str, ...] = (
    "python",
    "java",
    "cpp",
    "c",
    "csharp",
    "go",
    "javascript",
    "typescript",
    "php",
    "ruby",
    "rust",
    "swift",
)
LANG_SET: frozenset[str] = frozenset(LANGUAGES)


@dataclass
class PointRecord:
    id: str
    source: str                  # leetcode | codecomplex | kamyu | mbxp | rosetta | synthetic
    language: str                # one of LANGUAGES
    problem_id: Optional[str]
    solution_idx: Optional[int]
    code: str
    code_sha256: str
    label: str                   # one of POINT_LABELS
    raw_complexity: str
    tokens_bpe: int
    ast_nodes: int
    augmented_from: Optional[str] = None
    split: str = "train"         # train | val | test

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class PairRecord:
    pair_id: str
    language: str                # within-language pairs only; both sides share this language
    code_a: str
    code_b: str
    label_a: str
    label_b: str
    # Binary pair label post-rewrite: "same" | "A_faster" (= B is strictly slower).
    # The column name stays "ternary" for parquet schema compatibility with
    # earlier extractions; the *values* are binary now. Upstream canonicalization
    # guarantees tier_A <= tier_B so "B_faster" is never emitted.
    ternary: str
    same_problem: bool
    tokens_combined: int
    split: str = "train"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


POINT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source", pa.string()),
    ("language", pa.string()),
    ("problem_id", pa.string()),
    ("solution_idx", pa.int32()),
    ("code", pa.string()),
    ("code_sha256", pa.string()),
    ("label", pa.string()),
    ("raw_complexity", pa.string()),
    ("tokens_bpe", pa.int32()),
    ("ast_nodes", pa.int32()),
    ("augmented_from", pa.string()),
    ("split", pa.string()),
])

PAIR_SCHEMA = pa.schema([
    ("pair_id", pa.string()),
    ("language", pa.string()),
    ("code_a", pa.string()),
    ("code_b", pa.string()),
    ("label_a", pa.string()),
    ("label_b", pa.string()),
    ("ternary", pa.string()),
    ("same_problem", pa.bool_()),
    ("tokens_combined", pa.int32()),
    ("split", pa.string()),
])


# Raw intermediate record produced by the pipeline/02-04 parsers before normalization.
@dataclass
class RawRecord:
    source: str
    language: str
    problem_id: Optional[str]
    solution_idx: Optional[int]
    code: str
    raw_complexity: str
    extras: dict = field(default_factory=dict)
