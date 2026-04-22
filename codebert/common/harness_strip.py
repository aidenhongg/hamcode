"""Strip Codeforces-style I/O harness AND competitive-programming template
boilerplate from CodeComplex snippets.

CodeComplex snippets come from Codeforces submissions, which typically have:

  1. A fast-I/O template at the top:
        class FastIO(IOBase):
            def read(self):  ... while loop ...
            def readline(self): ... while loop ...

  2. Named helper functions:
        def read_int(): return int(input())
        def read_ints(): return list(map(int, input().split()))

  3. Lambda shortcuts:
        input = sys.stdin.readline
        LI = lambda: list(map(int, input().split()))

  4. Recursion-limit + alias setup:
        sys.setrecursionlimit(10**6)

  5. A `for _ in range(int(input()))` test-case scaffold wrapping the algo.

Without stripping, an O(1) algorithm "has" 20+ loops from FastIO alone,
flooding the complexity signal. We remove ALL of the above, keeping only:
  - the top-level algorithmic body
  - the `solve()` function (if present) — body kept intact
  - algorithmic loops, conditionals, data-structure ops
  - `print()` output statements (part of algorithmic behavior)

Fails open: if ast.parse or ast.unparse fails, returns the input unchanged.
"""

from __future__ import annotations

import ast
import re

# Well-known template class names (case-insensitive suffix match).
_TEMPLATE_CLASS_NAMES = frozenset({
    "fastio", "fastreader", "fastwriter", "fastinput", "fastoutput",
    "inputreader", "outputwriter", "bufferedreader", "bufferedwriter",
    "iowrapper", "streamreader", "streamwriter",
})

# Helper-function names that are almost always I/O wrappers, not algorithms.
_TEMPLATE_FUNC_NAMES = frozenset({
    "read", "readline", "readlines", "readint", "readints", "read_int",
    "read_ints", "read_str", "read_float", "read_floats", "readstr",
    "readfloat", "readfloats", "write", "writeline", "write_int",
    "input_as_int", "input_as_ints", "input_as_str",
    # extremely-short aliases common in CF templates
    "r", "ri", "rl", "rs", "rf", "rii", "rli", "rls", "li", "mi", "ii", "si",
    "rint", "rlist", "rmap",
})

# Common sys-level setup calls we can safely drop.
_DROP_SYS_CALLS = frozenset({
    "setrecursionlimit",
})


def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _call_name(node) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _is_int_input_call(node) -> bool:
    """int(input())"""
    if _call_name(node) != "int":
        return False
    if len(node.args) != 1:
        return False
    return _call_name(node.args[0]) == "input"


def _is_range_of_int_input(node) -> bool:
    """range(int(input()))"""
    if _call_name(node) != "range":
        return False
    if len(node.args) != 1:
        return False
    return _is_int_input_call(node.args[0])


def _is_range_of_known_var(node, known_vars: set[str]) -> bool:
    """range(T) where T is in known_vars"""
    if _call_name(node) != "range":
        return False
    if len(node.args) != 1:
        return False
    a = node.args[0]
    return isinstance(a, ast.Name) and a.id in known_vars


def _test_count_var(stmt) -> str | None:
    """Return var name if stmt is `T = int(input())`, else None."""
    if not isinstance(stmt, ast.Assign):
        return None
    if len(stmt.targets) != 1:
        return None
    tgt = stmt.targets[0]
    if not isinstance(tgt, ast.Name):
        return None
    if _is_int_input_call(stmt.value):
        return tgt.id
    return None


def _is_test_loop(stmt, known_vars: set[str]) -> bool:
    """Outer Codeforces test-case loop."""
    if not isinstance(stmt, ast.For):
        return False
    if _is_range_of_int_input(stmt.iter):
        return True
    if _is_range_of_known_var(stmt.iter, known_vars):
        return True
    return False


def _is_trivial_input_read(stmt) -> bool:
    """Assignment whose RHS is a pure input-reading expression."""
    if not isinstance(stmt, ast.Assign):
        return False
    dumped = ast.dump(stmt.value)
    if len(dumped) > 500:
        return False
    # Must mention input() somewhere in the RHS
    if "func=Name(id='input'" not in dumped:
        return False
    # Reject if RHS also contains meaningful structure (not pure I/O)
    # We keep the heuristic simple: any direct usage of input() -> trivial.
    return True


def _is_input_reading_loop(stmt) -> bool:
    """for _ in range(n): arr.append(int(input()))  or  [int(input()) for _ in range(n)]"""
    if not isinstance(stmt, ast.For):
        return False
    # Body must be exactly one append of an input-reading call
    if len(stmt.body) != 1:
        return False
    b0 = stmt.body[0]
    if not isinstance(b0, ast.Expr):
        return False
    if not isinstance(b0.value, ast.Call):
        return False
    # e.g. arr.append(int(input())) or similar
    dumped = ast.dump(b0.value)
    return "func=Name(id='input'" in dumped


def _is_stdin_alias(stmt) -> bool:
    """input = sys.stdin.readline / input = lambda: ... — drop this line."""
    if not isinstance(stmt, ast.Assign):
        return False
    tgt = stmt.targets[0] if stmt.targets else None
    if not _is_name(tgt, "input"):
        return False
    dumped = ast.dump(stmt.value)
    return (
        "stdin" in dumped
        or "readline" in dumped
        or isinstance(stmt.value, ast.Lambda)
    )


def _is_template_class(stmt) -> bool:
    """Drop well-known fast-I/O / buffered-I/O class definitions."""
    if not isinstance(stmt, ast.ClassDef):
        return False
    if stmt.name.lower() in _TEMPLATE_CLASS_NAMES:
        return True
    # Look at the base classes — inheriting from IOBase / RawIOBase / BufferedIOBase
    for base in stmt.bases:
        name = None
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute):
            name = base.attr
        if name and "IOBase" in name:
            return True
    return False


def _is_template_function(stmt) -> bool:
    """Drop helper functions whose body is pure I/O or matches known names."""
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if stmt.name.lower() in _TEMPLATE_FUNC_NAMES:
        return True
    # Short-body pure-I/O functions (typical helper wrappers).
    if len(stmt.body) == 1:
        b0 = stmt.body[0]
        # e.g. `def readint(): return int(input())`
        if isinstance(b0, ast.Return) and b0.value is not None:
            dumped = ast.dump(b0.value)
            if "func=Name(id='input'" in dumped and len(dumped) < 400:
                return True
    # Longer functions whose body is dominated by I/O primitives.
    dumped = ast.dump(stmt)
    if len(dumped) < 2000:
        n_io = dumped.count("id='input'") + dumped.count("'stdin'") + dumped.count("'stdout'")
        n_total = max(1, len(dumped) // 40)
        if n_io >= 2 and n_io / n_total > 0.25:
            return True
    return False


def _is_template_lambda_assign(stmt) -> bool:
    """Drop top-level lambda shortcuts: LI = lambda: list(map(int, input().split()))."""
    if not isinstance(stmt, ast.Assign):
        return False
    if not isinstance(stmt.value, ast.Lambda):
        return False
    dumped = ast.dump(stmt.value)
    return "func=Name(id='input'" in dumped


def _is_droppable_sys_call(stmt) -> bool:
    """sys.setrecursionlimit(...), sys.stdin / sys.stdout manipulation."""
    if not isinstance(stmt, ast.Expr):
        return False
    if not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "sys":
            if func.attr in _DROP_SYS_CALLS:
                return True
    if isinstance(func, ast.Name):
        if func.id in _DROP_SYS_CALLS:
            return True
    return False


def _strip_body(body: list) -> list:
    """Pass over a statement list; drop I/O, template classes/funcs, etc."""
    out = []
    for stmt in body:
        if _is_stdin_alias(stmt):
            continue
        if _is_template_lambda_assign(stmt):
            continue
        if _is_droppable_sys_call(stmt):
            continue
        if _is_template_class(stmt):
            continue
        if _is_template_function(stmt):
            continue
        if _is_trivial_input_read(stmt):
            continue
        if _is_input_reading_loop(stmt):
            continue
        # Recurse into bodies of compound statements (def/if/for/while) so that
        # an algorithmic for-loop containing I/O reads gets cleaned too.
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stmt.body = _strip_body(stmt.body)
        elif isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            stmt.body = _strip_body(stmt.body)
            if hasattr(stmt, "orelse"):
                stmt.orelse = _strip_body(stmt.orelse) if stmt.orelse else []
            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    h.body = _strip_body(h.body)
                stmt.finalbody = _strip_body(stmt.finalbody) if stmt.finalbody else []
        out.append(stmt)
    return out


def strip_harness(code: str) -> str:
    """Return `code` with Codeforces harness removed. Fails open."""
    if not code or not code.strip():
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Identify test-count variables: `T = int(input())`
    known_vars: set[str] = set()
    for stmt in tree.body:
        var = _test_count_var(stmt)
        if var is not None:
            known_vars.add(var)

    # Unwrap outer test-case loops. Also drop the preceding `T = int(input())`
    # assignment since its sole purpose was controlling the removed loop.
    new_body = []
    for stmt in tree.body:
        var = _test_count_var(stmt)
        if var is not None and var in known_vars:
            # Skip if any subsequent loop references it; otherwise keep.
            # Cheap check: scan the whole module once, drop if referenced.
            dumped_module = ast.dump(tree)
            if f"id='{var}'" in dumped_module.replace(
                f"targets=[Name(id='{var}'", "", 1
            ):
                continue
        if _is_test_loop(stmt, known_vars):
            new_body.extend(stmt.body)
            continue
        new_body.append(stmt)

    # Strip trivial I/O + input-reading loops recursively
    new_body = _strip_body(new_body)

    # If stripping emptied the body, bail out — the snippet was nearly all I/O.
    if not new_body:
        return code

    tree.body = new_body
    try:
        return ast.unparse(tree)
    except Exception:
        return code


# ---------- self-test harness ----------

_TEST_CASES: tuple[tuple[str, list[str], list[str]], ...] = (
    # (input, must_contain_after, must_not_contain_after)
    (
        # Classic t-test-case scaffold
        """T = int(input())
for _ in range(T):
    n = int(input())
    arr = list(map(int, input().split()))
    print(sum(arr))
""",
        ["print(sum(arr))"],
        ["for _ in range", "int(input())"],
    ),
    (
        # Range of int(input()) directly
        """for _ in range(int(input())):
    n = int(input())
    print(n * 2)
""",
        ["print(n * 2)"],
        ["for _ in range", "int(input())"],
    ),
    (
        # stdin alias
        """import sys
input = sys.stdin.readline
n = int(input())
arr = [i*i for i in range(n)]
print(sum(arr))
""",
        ["arr = [i * i for i in range(n)]", "print(sum(arr))"],
        ["sys.stdin.readline"],
    ),
    (
        # Algorithmic for-loop should NOT be stripped
        """def solve(arr):
    n = len(arr)
    result = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                result += 1
    return result
""",
        ["def solve", "for i in range(n)", "for j in range(i + 1, n)"],
        [],
    ),
    (
        # FastIO template — should be fully removed
        """import sys
from io import BytesIO, IOBase

class FastIO(IOBase):
    newlines = 0
    def __init__(self, file):
        self._fd = file.fileno()
        self._buffer = BytesIO()
    def read(self):
        while True:
            b = os.read(self._fd, 8192)
            if not b:
                break
            self._buffer.write(b)
        return self._buffer.read()

sys.setrecursionlimit(10**6)
input = sys.stdin.readline

def read_int():
    return int(input())

def read_ints():
    return list(map(int, input().split()))

LI = lambda: list(map(int, input().split()))

def solve():
    n = read_int()
    arr = read_ints()
    ans = 0
    for i in range(n):
        ans += arr[i] * arr[i]
    print(ans)

for _ in range(int(input())):
    solve()
""",
        ["def solve", "for i in range(n)", "print(ans)"],
        ["class FastIO", "def read_int", "def read_ints",
         "sys.setrecursionlimit", "LI = lambda", "sys.stdin.readline",
         "for _ in range(int(input()))"],
    ),
    (
        # Short-alias template helpers (R, RI, RL pattern very common in CF)
        """import sys
input = sys.stdin.readline
def R():
    return input()
def RI():
    return int(input())
def RL():
    return list(map(int, input().split()))
n = RI()
a = RL()
print(sorted(a))
""",
        ["print(sorted(a))"],
        ["def R()", "def RI()", "def RL()", "sys.stdin.readline"],
    ),
)


def _self_test() -> None:
    fails = 0
    for i, (src, must, mustnot) in enumerate(_TEST_CASES):
        out = strip_harness(src)
        ok = True
        for frag in must:
            if frag not in out:
                print(f"FAIL case {i}: missing {frag!r} in output:\n{out}\n")
                ok = False
        for frag in mustnot:
            if frag in out:
                print(f"FAIL case {i}: found forbidden {frag!r} in output:\n{out}\n")
                ok = False
        if not ok:
            fails += 1
    if fails == 0:
        print(f"harness_strip self-test OK ({len(_TEST_CASES)} cases)")
    else:
        raise SystemExit(f"{fails} harness_strip test failure(s)")


if __name__ == "__main__":
    _self_test()
