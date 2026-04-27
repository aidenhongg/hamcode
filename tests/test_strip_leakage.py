"""Unit tests for pipeline/05b_strip_leakage.py.

Covers:
  - Python `class Solution` unwrap (single + multiple methods, decorators,
    `self`-drop, kamyu-style `Solution2(object)`, decorated_definition).
  - Non-Solution Python classes left alone.
  - Ruby `class Solution` unwrap (no `self` drop).
  - Languages other than Python/Ruby: wrapper preserved (per Q1).
  - Comment stripping across languages: Time:/Space: headers, bare `O(n)`
    line, `complexity` keyword, end-of-line comment, in-string preservation.
  - Stripping that breaks syntax → recognized so the stage can drop the row.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_strip_module():
    spec = importlib.util.spec_from_file_location(
        "strip_leakage", REPO / "pipeline" / "05b_strip_leakage.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def strip():
    return _load_strip_module()


# ---------------------------------------------------------------------------
# Python wrapper unwrap
# ---------------------------------------------------------------------------

def test_python_solution_single_method_unwrapped(strip):
    code = (
        "class Solution:\n"
        "    def two_sum(self, nums, target):\n"
        "        return [0, 1]\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "class Solution" not in new
    assert "def two_sum(nums, target)" in new
    assert "self" not in new


def test_python_solution_multiple_methods_unwrapped(strip):
    code = (
        "class Solution:\n"
        "    def foo(self, x):\n"
        "        return x\n"
        "\n"
        "    def bar(self, y, z):\n"
        "        return y + z\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "def foo(x):" in new
    assert "def bar(y, z):" in new


def test_python_solution_with_object_base(strip):
    code = (
        "class Solution(object):\n"
        "    def foo(self):\n"
        "        return 1\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "def foo():" in new


def test_python_solution_with_decorator(strip):
    code = (
        "class Solution:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        return x\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "@staticmethod" in new
    assert "def foo(x):" in new


def test_python_solution_lru_cache_decorator_drops_self(strip):
    code = (
        "import functools\n"
        "class Solution:\n"
        "    @functools.lru_cache\n"
        "    def memo(self, n):\n"
        "        return n\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "@functools.lru_cache" in new
    assert "def memo(n):" in new


def test_python_solution_numbered_variant(strip):
    """kamyu sometimes emits Solution2 / Solution3 for alternate approaches."""
    code = (
        "class Solution2(object):\n"
        "    def foo(self):\n"
        "        return 1\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "def foo():" in new


def test_python_treenode_helper_preserved(strip):
    """Helper data structures with non-Solution names are left alone."""
    code = (
        "class TreeNode:\n"
        "    def __init__(self, val):\n"
        "        self.val = val\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is False
    assert "class TreeNode" in new
    assert "self.val" in new


def test_python_treenode_alongside_solution(strip):
    code = (
        "class TreeNode:\n"
        "    def __init__(self, val):\n"
        "        self.val = val\n"
        "\n"
        "class Solution:\n"
        "    def height(self, root):\n"
        "        return 0\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    assert "class TreeNode" in new
    assert "class Solution" not in new
    assert "def height(root):" in new


def test_python_no_class_passes_through(strip):
    code = "def foo(x):\n    return x + 1\n"
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is False
    assert audit["n_comments_stripped"] == 0
    assert new == code


# ---------------------------------------------------------------------------
# Ruby wrapper unwrap
# ---------------------------------------------------------------------------

def test_ruby_solution_unwrapped(strip):
    code = (
        "class Solution\n"
        "  def two_sum(nums, target)\n"
        "    nums\n"
        "  end\n"
        "end\n"
    )
    new, audit = strip.strip_record(code, "ruby")
    assert audit["was_unwrapped"] is True
    assert "class Solution" not in new
    assert "def two_sum" in new


def test_ruby_non_solution_class_preserved(strip):
    code = (
        "class TreeNode\n"
        "  def initialize(val)\n"
        "    @val = val\n"
        "  end\n"
        "end\n"
    )
    new, audit = strip.strip_record(code, "ruby")
    assert audit["was_unwrapped"] is False
    assert "class TreeNode" in new


# ---------------------------------------------------------------------------
# Other-language wrappers stay intact (Q1)
# ---------------------------------------------------------------------------

def test_java_solution_NOT_unwrapped(strip):
    code = (
        "class Solution {\n"
        "    public int twoSum(int[] nums) {\n"
        "        return 0;\n"
        "    }\n"
        "}\n"
    )
    new, audit = strip.strip_record(code, "java")
    assert audit["was_unwrapped"] is False
    assert "class Solution" in new


def test_cpp_solution_NOT_unwrapped(strip):
    code = (
        "class Solution {\n"
        "public:\n"
        "    int twoSum(int x) { return x; }\n"
        "};\n"
    )
    new, audit = strip.strip_record(code, "cpp")
    assert audit["was_unwrapped"] is False
    assert "class Solution" in new


def test_rust_impl_solution_NOT_unwrapped(strip):
    code = (
        "impl Solution {\n"
        "    pub fn two_sum(nums: Vec<i32>) -> Vec<i32> {\n"
        "        nums\n"
        "    }\n"
        "}\n"
    )
    new, audit = strip.strip_record(code, "rust")
    assert audit["was_unwrapped"] is False
    assert "impl Solution" in new


# ---------------------------------------------------------------------------
# Complexity-comment stripping
# ---------------------------------------------------------------------------

def test_strip_kamyu_time_header_python(strip):
    code = (
        "# Time:  O(n)\n"
        "# Space: O(n)\n"
        "def foo():\n"
        "    return 1\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["n_comments_stripped"] == 2
    assert "Time:" not in new
    assert "Space:" not in new
    assert "def foo():" in new


def test_strip_kamyu_time_header_java(strip):
    code = (
        "// Time:  O(n)\n"
        "// Space: O(n)\n"
        "class A {\n"
        "    int f(int x) { return x; }\n"
        "}\n"
    )
    new, audit = strip.strip_record(code, "java")
    assert audit["n_comments_stripped"] == 2
    assert "Time:" not in new


def test_strip_bare_big_o_comment(strip):
    code = "def foo():\n    # O(n)\n    return 1\n"
    new, audit = strip.strip_record(code, "python")
    assert audit["n_comments_stripped"] == 1
    assert "O(n)" not in new


def test_strip_complexity_keyword(strip):
    code = "# this function has linear complexity\ndef foo():\n    return 1\n"
    new, audit = strip.strip_record(code, "python")
    assert audit["n_comments_stripped"] == 1
    assert "complexity" not in new.lower()


def test_eol_comment_stripped_keeps_code(strip):
    code = "def foo(x):\n    return x  # O(n) operation\n"
    new, audit = strip.strip_record(code, "python")
    assert audit["n_comments_stripped"] == 1
    # Code line preserved, comment + leading spaces gone
    assert "return x\n" in new
    assert "O(n)" not in new


def test_string_literal_with_big_o_preserved(strip):
    """Tree-sitter scoping ensures `O(n)` inside a string isn't matched."""
    code = (
        "def f():\n"
        "    print(\"runs in O(n) time\")\n"
        "    return 1\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["n_comments_stripped"] == 0
    assert 'print("runs in O(n) time")' in new


def test_block_comment_with_complexity_stripped_cpp(strip):
    code = (
        "/* Time: O(n) */\n"
        "int f(int x) { return x; }\n"
    )
    new, audit = strip.strip_record(code, "cpp")
    assert audit["n_comments_stripped"] >= 1
    assert "Time:" not in new


def test_non_complexity_comment_preserved(strip):
    code = (
        "# this is a regular comment about types\n"
        "def foo():\n"
        "    return 1\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["n_comments_stripped"] == 0
    assert "regular comment" in new


# ---------------------------------------------------------------------------
# Round-trip syntax validity
# ---------------------------------------------------------------------------

def test_unwrapped_python_still_parses(strip):
    """The unwrap must produce code that the tree-sitter syntax check accepts."""
    from common.parsers import syntax_ok
    code = (
        "class Solution:\n"
        "    def foo(self, x):\n"
        "        return x + 1\n"
        "\n"
        "    def bar(self, y):\n"
        "        return y * 2\n"
    )
    new, _ = strip.strip_record(code, "python")
    assert syntax_ok("python", new)


def test_unwrapped_ruby_still_parses(strip):
    from common.parsers import syntax_ok
    code = (
        "class Solution\n"
        "  def two_sum(nums)\n"
        "    nums\n"
        "  end\n"
        "end\n"
    )
    new, _ = strip.strip_record(code, "ruby")
    assert syntax_ok("ruby", new)


# ---------------------------------------------------------------------------
# Combined: wrapper + comment in one record
# ---------------------------------------------------------------------------

def test_combined_solution_with_complexity_header(strip):
    code = (
        "# Time:  O(n)\n"
        "# Space: O(1)\n"
        "class Solution(object):\n"
        "    def two_sum(self, nums, target):\n"
        "        # O(n) walk\n"
        "        return [0, 1]\n"
    )
    new, audit = strip.strip_record(code, "python")
    assert audit["was_unwrapped"] is True
    # The pre-class headers + the in-method comment all qualify; tree-sitter
    # comment nodes outside the class get explicitly stripped, while the
    # in-class one gets dropped along with the class body's non-method
    # statements (we keep only function_definition / decorated_definition).
    assert "Time:" not in new
    assert "Space:" not in new
    assert "O(n)" not in new
    assert "def two_sum(nums, target):" in new


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
