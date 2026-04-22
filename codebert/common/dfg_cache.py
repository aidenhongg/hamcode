"""On-disk DFG cache keyed by code SHA256.

Tree-sitter parsing + DFG extraction is deterministic and accounts for a big
chunk of data-loading time. We cache the intermediate representation (token
list + edges + variable indices) as pickled blobs; data.py calls get()/put().

Cache directory defaults to ~/.cache/codebert/dfg but can be overridden.
Single-writer assumption — we rely on atomic POSIX rename on Linux/RunPod.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


class DFGCache:
    def __init__(self, root: str | os.PathLike | None = None) -> None:
        if root is None:
            root = Path.home() / ".cache" / "codebert" / "dfg"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # Fan out first 2 chars to avoid directory-scan blowup (>1M files in one dir).
        return self.root / key[:2] / f"{key}.pkl"

    def get(self, code: str) -> Any | None:
        p = self._path(self.key(code))
        if not p.exists():
            return None
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError):
            return None

    def put(self, code: str, value: Any) -> None:
        p = self._path(self.key(code))
        p.parent.mkdir(parents=True, exist_ok=True)
        # atomic write
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def __contains__(self, code: str) -> bool:
        return self._path(self.key(code)).exists()
