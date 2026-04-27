"""Smoke test for pipeline/05b_strip_leakage.py invoked end-to-end.

Builds a tiny normalized.jsonl with mixed sources/languages, runs the
strip stage as a subprocess, and asserts the output jsonl + audit log
look right.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def workdir(tmp_path):
    """Return a tmp dir with a crafted normalized/combined.jsonl."""
    in_path = tmp_path / "in" / "normalized" / "combined.jsonl"
    in_path.parent.mkdir(parents=True)
    rows = [
        # 1. Python class Solution + kamyu time header
        {
            "id": "r1", "source": "kamyu", "language": "python",
            "problem_id": "p1", "code": (
                "# Time: O(n)\n"
                "# Space: O(1)\n"
                "class Solution(object):\n"
                "    def two_sum(self, nums, target):\n"
                "        return [0, 1]\n"
            ),
            "label": "O(n)", "raw_complexity": "O(n)",
        },
        # 2. Java Solution wrapper retained (per Q1)
        {
            "id": "r2", "source": "kamyu", "language": "java",
            "problem_id": "p2", "code": (
                "// Time: O(n)\n"
                "class Solution {\n"
                "    public int f(int x) { return x; }\n"
                "}\n"
            ),
            "label": "O(n)", "raw_complexity": "O(n)",
        },
        # 3. String literal with O(n) preserved
        {
            "id": "r3", "source": "leetcode", "language": "python",
            "problem_id": "p3", "code": (
                "def f():\n"
                "    return \"runs in O(n) time\"\n"
            ),
            "label": "O(1)", "raw_complexity": "O(1)",
        },
        # 4. No-leakage row identical to input
        {
            "id": "r4", "source": "synthetic", "language": "python",
            "problem_id": "p4", "code": (
                "def add(a, b):\n"
                "    return a + b\n"
            ),
            "label": "O(1)", "raw_complexity": "O(1)",
        },
    ]
    with in_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def test_strip_stage_subprocess(workdir):
    in_path = workdir / "in" / "normalized" / "combined.jsonl"
    out_path = workdir / "stripped" / "combined.jsonl"
    audit_path = workdir / "audit" / "strip_log.jsonl"
    fail_path = workdir / "audit" / "strip_failures.jsonl"

    proc = subprocess.run(
        [sys.executable, str(REPO / "pipeline" / "05b_strip_leakage.py"),
         "--in_path", str(in_path),
         "--out", str(out_path),
         "--audit_log", str(audit_path),
         "--fail_log", str(fail_path)],
        cwd=str(REPO),
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"05b stage failed:\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
    )

    out_rows = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(out_rows) == 4, f"expected 4 rows, got {len(out_rows)}: {out_rows}"

    by_id = {r["id"]: r for r in out_rows}

    # 1. Python row: unwrapped + comments stripped
    r1 = by_id["r1"]
    assert "class Solution" not in r1["code"]
    assert "def two_sum(nums, target)" in r1["code"]
    assert "Time:" not in r1["code"]
    assert r1.get("was_stripped") is True

    # 2. Java row: wrapper preserved, comment stripped
    r2 = by_id["r2"]
    assert "class Solution" in r2["code"]
    assert "Time:" not in r2["code"]
    assert r2.get("was_stripped") is True

    # 3. String literal preserved
    r3 = by_id["r3"]
    assert "O(n)" in r3["code"]
    # was_stripped may be False since nothing actually changed
    assert r3.get("was_stripped") is False

    # 4. No-leakage row identical to input
    r4 = by_id["r4"]
    assert r4["code"] == "def add(a, b):\n    return a + b\n"
    assert r4.get("was_stripped") is False

    # Audit log: per-row records exist
    audit_lines = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(audit_lines) == 4
    # No syntax failures expected
    fail_text = fail_path.read_text(encoding="utf-8") if fail_path.exists() else ""
    assert fail_text.strip() == "", f"unexpected strip failures: {fail_text}"

    # Console summary mentions language counts
    assert "[05b]" in proc.stdout
    assert "in=4" in proc.stdout
    assert "out=4" in proc.stdout
    assert "fail=0" in proc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
