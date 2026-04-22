"""Normalize raw complexity strings to one of 11 canonical labels, or reject.

Pipeline:
    pre_canonicalize -> strip O(...) wrapper -> variable remap -> pattern match.

Reject policy: if no pattern matches (or the expression is ambiguous like
`O(k * n)` with unclear `k`, or `O(m + n log n)` that straddles tiers), return
None. Callers log rejects to audit/normalize_rejects.jsonl.
"""

from __future__ import annotations

import re

from .labels import POINT_LABELS

_RESERVED = {"log", "ln", "sqrt", "alpha", "o"}


def pre_canonicalize(s: str) -> str:
    """Strip LaTeX, unify operators, lowercase, collapse whitespace."""
    if not s:
        return ""
    s = s.strip()
    # Strip LaTeX dollars, $$ ... $$
    s = s.replace("$$", "$").replace("$", "")
    s = s.rstrip(".,;:")
    # LaTeX commands
    s = re.sub(r"\\times\b", "*", s)
    s = re.sub(r"\\cdot\b", "*", s)
    s = re.sub(r"\\log\b", "log", s)
    s = re.sub(r"\\ln\b", "log", s)
    s = re.sub(r"\\sqrt\b", "sqrt", s)
    s = re.sub(r"\\alpha\b", "alpha", s)
    s = re.sub(r"\{|\}", "", s)
    # bare `ln` (not LaTeX) -> log
    s = re.sub(r"\bln\b", "log", s)
    # Unicode
    s = s.replace("·", "*").replace("×", "*").replace("⋅", "*")
    # Exponent
    s = s.replace("^", "**")
    # log_2 -> log  (base doesn't matter for asymptotics)
    s = re.sub(r"\blog_?\s*\d+", "log", s)
    # lowercase
    s = s.lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_inner(s: str) -> str:
    """Return the inside of the outermost O(...), or the string itself."""
    m = re.match(r"^\s*o\s*\((.+)\)\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def _word_var_replace(expr: str) -> str:
    # Word-length variable names common in LeetCode: rows/cols, height/width, s/t for strings
    expr = re.sub(r"\brows?\b", "m", expr)
    expr = re.sub(r"\bcols?\b|\bcolumns?\b", "n", expr)
    expr = re.sub(r"\bheight\b", "m", expr)
    expr = re.sub(r"\bwidth\b", "n", expr)
    expr = re.sub(r"\blen\b", "n", expr)
    return expr


def normalize_variables(expr: str) -> str:
    """Canonicalize variable letters to {m, n}. Keep n dominant; map others to m."""
    expr = _word_var_replace(expr)

    letters = set(re.findall(r"\b([a-z])\b", expr)) - _RESERVED - {"c", "k"}
    # `c` and `k` are conventional constants; drop them from "variables" consideration.
    # Exception: `c^n` → exponential (handled by pattern below).

    if "n" in letters:
        others = sorted(letters - {"n"})
        if len(others) >= 1:
            # Map the first other letter to m; if there's a second, we'll fail to match.
            expr = re.sub(rf"\b{re.escape(others[0])}\b", "m", expr)
            for o in others[1:]:
                # Tag remaining letters so they become literal "?" and fail matching.
                expr = re.sub(rf"\b{re.escape(o)}\b", "?", expr)
    elif len(letters) == 1:
        only = next(iter(letters))
        expr = re.sub(rf"\b{re.escape(only)}\b", "n", expr)
    elif len(letters) == 2:
        a, b = sorted(letters)
        expr = re.sub(rf"\b{re.escape(a)}\b", "m", expr)
        expr = re.sub(rf"\b{re.escape(b)}\b", "n", expr)
    # 0 letters: maybe a pure constant like "1"; leave alone.
    # 3+ letters: leave unmatched -> will fail pattern match -> reject.
    return expr


# Patterns on the normalized inner expression (post `normalize_variables`, post
# `_tighten`). First match wins. Lists are ordered to avoid shadowing.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # === overrides ===
    (re.compile(r"^alpha\s*\(\s*n\s*\)$"), "O(1)"),
    (re.compile(r"^alpha\s*\*\s*n$"), "O(1)"),

    # === multi-variable (checked before single-var to avoid O(n) swallowing) ===
    (re.compile(r"^\(\s*m\s*\+\s*n\s*\)\s*\*\s*log\s*\(\s*m\s*\+\s*n\s*\)$"), "O((m+n) log(m+n))"),
    (re.compile(r"^\(\s*n\s*\+\s*m\s*\)\s*\*\s*log\s*\(\s*n\s*\+\s*m\s*\)$"), "O((m+n) log(m+n))"),

    (re.compile(r"^m\s*\*\s*log\s*\(?\s*n\s*\)?$"), "O(m log n)"),
    (re.compile(r"^n\s*\*\s*log\s*\(?\s*m\s*\)?$"), "O(m log n)"),
    (re.compile(r"^log\s*\(?\s*m\s*\)?\s*\*\s*n$"), "O(m log n)"),
    (re.compile(r"^log\s*\(?\s*n\s*\)?\s*\*\s*m$"), "O(m log n)"),

    (re.compile(r"^m\s*\*\s*n$"), "O(m*n)"),
    (re.compile(r"^n\s*\*\s*m$"), "O(m*n)"),

    (re.compile(r"^m\s*\+\s*n$"), "O(m+n)"),
    (re.compile(r"^n\s*\+\s*m$"), "O(m+n)"),

    # === single-variable ===
    # Exponential / factorial (check before n^k so 2^n isn't confused)
    (re.compile(r"^2\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^\d+\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^c\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^n\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^n\s*!$"), "exponential"),
    (re.compile(r"^n\s*\*\s*n\s*!$"), "exponential"),
    (re.compile(r"^\(\s*n\s*-\s*\d+\s*\)\s*!$"), "exponential"),

    # O(n^3)
    (re.compile(r"^n\s*\*\*\s*3$"), "O(n^3)"),
    (re.compile(r"^n\s*\*\s*n\s*\*\s*n$"), "O(n^3)"),

    # O(n^2)
    (re.compile(r"^n\s*\*\*\s*2$"), "O(n^2)"),
    (re.compile(r"^n\s*\*\s*n$"), "O(n^2)"),

    # O(n log n)   -- also log(n!) = n log n
    (re.compile(r"^n\s*\*\s*log\s*\(?\s*n\s*\)?$"), "O(n log n)"),
    (re.compile(r"^n\s*log\s*\(?\s*n\s*\)?$"), "O(n log n)"),
    (re.compile(r"^log\s*\(?\s*n\s*\)?\s*\*\s*n$"), "O(n log n)"),
    (re.compile(r"^log\s*\(\s*n\s*!\s*\)$"), "O(n log n)"),

    # O(log n)
    (re.compile(r"^log\s*\(?\s*n\s*\)?$"), "O(log n)"),

    # O(n)
    (re.compile(r"^n$"), "O(n)"),
    (re.compile(r"^n\s*\+\s*\d+$"), "O(n)"),
    (re.compile(r"^\d+\s*\*\s*n$"), "O(n)"),
    (re.compile(r"^n\s*\+\s*k$"), "O(n)"),      # counting sort: O(n+k) with k=alphabet
    (re.compile(r"^k\s*\*\s*n$"), "O(n)"),      # k treated as constant
    (re.compile(r"^n\s*\*\s*k$"), "O(n)"),

    # O(1)
    (re.compile(r"^\d+$"), "O(1)"),
]


def _tighten(expr: str) -> str:
    """Remove whitespace around operators for cleaner matching.

    `**` must be protected from the single-`*` regex — otherwise `n**2` becomes
    `n * * 2` and pattern matching breaks.
    """
    pow_marker = "\x00POW\x00"
    expr = expr.replace("**", pow_marker)
    expr = re.sub(r"\s*\+\s*", " + ", expr)
    expr = re.sub(r"\s*\*\s*", " * ", expr)
    expr = expr.replace(pow_marker, "**")
    expr = re.sub(r"\(\s+", "(", expr)
    expr = re.sub(r"\s+\)", ")", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def normalize(raw: str) -> str | None:
    """Return canonical label (one of POINT_LABELS) or None (reject)."""
    if not raw:
        return None
    s = pre_canonicalize(raw)
    if not s:
        return None
    inner = extract_inner(s)
    # Reject sqrt: not in our class set and we'd rather have holes than noise.
    if "sqrt" in inner:
        return None
    # Reject obvious garbage
    if "?" in inner:
        return None
    inner = normalize_variables(inner)
    inner = _tighten(inner)

    for pat, lab in _PATTERNS:
        if pat.match(inner):
            assert lab in POINT_LABELS, f"bug: pattern produced unknown label {lab}"
            return lab
    return None


# ---------- Self-test harness (run: `python -m common.normalizer`) ----------

_CASES_MATCH: tuple[tuple[str, str], ...] = (
    ("$O(1)$", "O(1)"),
    ("$O(n)$", "O(n)"),
    ("$O(\\log n)$", "O(log n)"),
    ("$O(n \\log n)$", "O(n log n)"),
    ("$O(n^2)$", "O(n^2)"),
    ("$O(n^3)$", "O(n^3)"),
    ("$O(2^n)$", "exponential"),
    ("$O(n!)$", "exponential"),
    ("$O(n \\times n!)$", "exponential"),
    ("$O(m + n)$", "O(m+n)"),
    ("$O(n + m)$", "O(m+n)"),
    ("$O(m \\times n)$", "O(m*n)"),
    ("$O(n \\times m)$", "O(m*n)"),
    ("$O(m \\times \\log n)$", "O(m log n)"),
    ("$O(n \\times \\log m)$", "O(m log n)"),
    ("$O((m + n) \\times \\log(m + n))$", "O((m+n) log(m+n))"),
    ("O(log n)", "O(log n)"),
    ("O(log_2 n)", "O(log n)"),
    ("O(ln n)", "O(log n)"),
    ("O(log(n!))", "O(n log n)"),
    ("O(rows * cols)", "O(m*n)"),
    ("O(k * n)", "O(n)"),               # `k` treated as constant per _RESERVED/exclude
    ("O(n + k)", "O(n)"),
)

_CASES_REJECT: tuple[str, ...] = (
    "$O(\\sqrt{n})$",
    "$O(\\alpha(n))$ is handled — but if it's malformed, reject",
    "O(n * log n * m)",     # too many terms
    "O(m + n * log n)",     # straddles tiers
    "O(a * b * c)",          # 3+ vars after remap
    "",
    None,    # type: ignore
)


def _self_test() -> None:
    fails = 0
    for raw, expected in _CASES_MATCH:
        got = normalize(raw)
        if got != expected:
            print(f"FAIL match: {raw!r} -> {got!r}, expected {expected!r}")
            fails += 1
    # Allow α(n) case to optionally match
    for raw, expected in (("$O(\\alpha(n))$", "O(1)"),):
        got = normalize(raw)
        if got != expected:
            print(f"WARN alpha: {raw!r} -> {got!r}, expected {expected!r}")
    for raw in _CASES_REJECT:
        got = normalize(raw)
        if got is not None:
            print(f"FAIL reject: {raw!r} -> {got!r}, expected None")
            fails += 1
    if fails == 0:
        print(f"normalize self-test OK ({len(_CASES_MATCH)} match + {len(_CASES_REJECT)} reject)")
    else:
        raise SystemExit(f"{fails} normalizer self-test failure(s)")


if __name__ == "__main__":
    _self_test()
