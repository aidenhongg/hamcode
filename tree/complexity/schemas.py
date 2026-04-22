"""Point-only record schema, pyarrow compatible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pyarrow as pa


@dataclass
class PointRecord:
    id: str
    source: str                  # codecomplex | leetcode | codeforces
    problem_id: Optional[str]
    solution_idx: Optional[int]
    code: str
    code_sha256: str
    label: str                   # one of POINT_LABELS
    raw_complexity: str
    origin: str                  # "dataset" | "comment" (how the label was obtained)
    ast_nodes: int
    split: str = "train"         # train | val | test

    def to_dict(self) -> dict:
        return self.__dict__.copy()


POINT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("source", pa.string()),
    ("problem_id", pa.string()),
    ("solution_idx", pa.int32()),
    ("code", pa.string()),
    ("code_sha256", pa.string()),
    ("label", pa.string()),
    ("raw_complexity", pa.string()),
    ("origin", pa.string()),
    ("ast_nodes", pa.int32()),
    ("split", pa.string()),
])
