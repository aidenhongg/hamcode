"""Normalize raw complexity strings to one of 11 canonical labels, or reject.

Pipeline:
    pre_canonicalize -> strip O(...) wrapper -> variable remap -> pattern match.

Mirrors codebert/common/normalizer.py. Kept in sync manually; the self-test at
the bottom validates parity with the codebert cases.
"""

from __future__ import annotations

import re

from .labels import POINT_LABELS

_RESERVED = {"log", "ln", "sqrt", "alpha", "o"}


def pre_canonicalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("$$", "$").replace("$", "")
    s = s.rstrip(".,;:")
    s = re.sub(r"\\times\b", "*", s)
    s = re.sub(r"\\cdot\b", "*", s)
    s = re.sub(r"\\log\b", "log", s)
    s = re.sub(r"\\ln\b", "log", s)
    s = re.sub(r"\\sqrt\b", "sqrt", s)
    s = re.sub(r"\\alpha\b", "alpha", s)
    s = re.sub(r"\\max\b", "max", s)
    s = re.sub(r"\\min\b", "min", s)
    # \textit{name} / \mathit{name} / \mathrm{name} — unwrap to just `name`
    s = re.sub(r"\\(?:textit|mathit|mathrm|text)\{([^}]+)\}", r"\1", s)
    # |\Sigma| and variants -> a constant marker. Upper or lower, with spaces.
    s = re.sub(r"\|\s*\\?sigma\s*\|", "c", s, flags=re.IGNORECASE)
    s = re.sub(r"\|[a-z]\|", "c", s, flags=re.IGNORECASE)  # |x|, |S|, etc. -> const
    s = re.sub(r"\{|\}", "", s)
    s = re.sub(r"\bln\b", "log", s)
    s = s.replace("·", "*").replace("×", "*").replace("⋅", "*")
    s = s.replace("^", "**")
    s = re.sub(r"\blog_?\s*\d+", "log", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_inner(s: str) -> str:
    m = re.match(r"^\s*o\s*\((.+)\)\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def _word_var_replace(expr: str) -> str:
    # Matrix dims (common in LeetCode problem editorials)
    expr = re.sub(r"\brows?\b", "m", expr)
    expr = re.sub(r"\bcols?\b|\bcolumns?\b", "n", expr)
    expr = re.sub(r"\bheight\b", "m", expr)
    expr = re.sub(r"\bwidth\b", "n", expr)
    expr = re.sub(r"\blen\b", "n", expr)
    # LeetCode-common multi-letter input names
    expr = re.sub(r"\bnums?\b", "n", expr)
    expr = re.sub(r"\btarget\b", "n", expr)
    expr = re.sub(r"\bval(?:ue)?\b", "n", expr)
    expr = re.sub(r"\bsize\b", "n", expr)
    # Named constants (bounded alphabets / limits) -> single 'c' so pattern matching
    # can use bounded-constant patterns below.
    expr = re.sub(r"\bsigma\b", "c", expr)
    return expr


def normalize_variables(expr: str) -> str:
    expr = _word_var_replace(expr)
    letters = set(re.findall(r"\b([a-z])\b", expr)) - _RESERVED - {"c", "k"}
    if "n" in letters:
        others = sorted(letters - {"n"})
        if len(others) >= 1:
            expr = re.sub(rf"\b{re.escape(others[0])}\b", "m", expr)
            for o in others[1:]:
                expr = re.sub(rf"\b{re.escape(o)}\b", "?", expr)
    elif len(letters) == 1:
        only = next(iter(letters))
        expr = re.sub(rf"\b{re.escape(only)}\b", "n", expr)
    elif len(letters) == 2:
        a, b = sorted(letters)
        expr = re.sub(rf"\b{re.escape(a)}\b", "m", expr)
        expr = re.sub(rf"\b{re.escape(b)}\b", "n", expr)
    return expr


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # === inverse Ackermann (treated as constant) ===
    (re.compile(r"^alpha\s*\(\s*n\s*\)$"), "O(1)"),
    (re.compile(r"^alpha\s*\*\s*n$"), "O(n)"),
    (re.compile(r"^n\s*\*\s*alpha\s*\(\s*n\s*\)$"), "O(n)"),

    # === sum-of-logs (m log m + n log n) ≈ (m+n) log(m+n) ===
    (re.compile(r"^m\s*\*?\s*log\s*\(?\s*m\s*\)?\s*\+\s*n\s*\*?\s*log\s*\(?\s*n\s*\)?$"),
     "O((m+n) log(m+n))"),
    (re.compile(r"^n\s*\*?\s*log\s*\(?\s*n\s*\)?\s*\+\s*m\s*\*?\s*log\s*\(?\s*m\s*\)?$"),
     "O((m+n) log(m+n))"),

    # === explicit (m+n) log(m+n) ===
    (re.compile(r"^\(\s*m\s*\+\s*n\s*\)\s*\*?\s*log\s*\(\s*m\s*\+\s*n\s*\)$"), "O((m+n) log(m+n))"),
    (re.compile(r"^\(\s*n\s*\+\s*m\s*\)\s*\*?\s*log\s*\(\s*n\s*\+\s*m\s*\)$"), "O((m+n) log(m+n))"),

    # === O(m log n) ===
    (re.compile(r"^m\s*\*?\s*log\s*\(?\s*n\s*\)?$"), "O(m log n)"),
    (re.compile(r"^n\s*\*?\s*log\s*\(?\s*m\s*\)?$"), "O(m log n)"),
    (re.compile(r"^log\s*\(?\s*m\s*\)?\s*\*?\s*n$"), "O(m log n)"),
    (re.compile(r"^log\s*\(?\s*n\s*\)?\s*\*?\s*m$"), "O(m log n)"),

    # === max/min(m, n) ≈ m+n tier ===
    (re.compile(r"^max\s*\(\s*m\s*,\s*n\s*\)$"), "O(m+n)"),
    (re.compile(r"^max\s*\(\s*n\s*,\s*m\s*\)$"), "O(m+n)"),
    (re.compile(r"^min\s*\(\s*m\s*,\s*n\s*\)$"), "O(m+n)"),
    (re.compile(r"^min\s*\(\s*n\s*,\s*m\s*\)$"), "O(m+n)"),

    # === O(m*n) ===
    (re.compile(r"^m\s*\*\s*n$"), "O(m*n)"),
    (re.compile(r"^n\s*\*\s*m$"), "O(m*n)"),

    # === O(m+n) ===
    (re.compile(r"^m\s*\+\s*n$"), "O(m+n)"),
    (re.compile(r"^n\s*\+\s*m$"), "O(m+n)"),

    # === exponential × polynomial in n — still exponential ===
    (re.compile(r"^n\s*\*\s*2\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^2\s*\*\*\s*n\s*\*\s*n$"), "exponential"),
    (re.compile(r"^n\s*\*\*\s*\d+\s*\*\s*2\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^2\s*\*\*\s*n\s*\*\s*n\s*\*\*\s*\d+$"), "exponential"),

    # === exponential / factorial ===
    (re.compile(r"^2\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^\d+\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^c\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^n\s*\*\*\s*n$"), "exponential"),
    (re.compile(r"^n\s*!$"), "exponential"),
    (re.compile(r"^n\s*\*\s*n\s*!$"), "exponential"),
    (re.compile(r"^\(\s*n\s*-\s*\d+\s*\)\s*!$"), "exponential"),

    # === O(n^3) ===
    (re.compile(r"^n\s*\*\*\s*3$"), "O(n^3)"),
    (re.compile(r"^n\s*\*\s*n\s*\*\s*n$"), "O(n^3)"),

    # === O(n^2) ===
    (re.compile(r"^n\s*\*\*\s*2$"), "O(n^2)"),
    (re.compile(r"^n\s*\*\s*n$"), "O(n^2)"),

    # === O(n log n) ===
    (re.compile(r"^n\s*\*\s*log\s*\(?\s*n\s*\)?$"), "O(n log n)"),
    (re.compile(r"^n\s*log\s*\(?\s*n\s*\)?$"), "O(n log n)"),
    (re.compile(r"^log\s*\(?\s*n\s*\)?\s*\*\s*n$"), "O(n log n)"),
    (re.compile(r"^log\s*\(\s*n\s*!\s*\)$"), "O(n log n)"),

    # === O(log n) ===
    (re.compile(r"^log\s*\(?\s*n\s*\)?$"), "O(log n)"),

    # === O(n) ===
    (re.compile(r"^n$"), "O(n)"),
    (re.compile(r"^n\s*\+\s*\d+$"), "O(n)"),
    (re.compile(r"^\d+\s*\*\s*n$"), "O(n)"),
    (re.compile(r"^n\s*\+\s*k$"), "O(n)"),
    (re.compile(r"^k\s*\*\s*n$"), "O(n)"),
    (re.compile(r"^n\s*\*\s*k$"), "O(n)"),
    (re.compile(r"^n\s*\+\s*c$"), "O(n)"),       # O(n + |Σ|) after Σ→c
    (re.compile(r"^c\s*\+\s*n$"), "O(n)"),
    (re.compile(r"^n\s*\*\s*c$"), "O(n)"),       # O(n × |Σ|)
    (re.compile(r"^c\s*\*\s*n$"), "O(n)"),
    (re.compile(r"^n\s*\*\s*\d+$"), "O(n)"),

    # === O(1) ===
    (re.compile(r"^\d+$"), "O(1)"),
    (re.compile(r"^k$"), "O(1)"),              # bare constant
    (re.compile(r"^c$"), "O(1)"),              # bare constant
    (re.compile(r"^k\s*\*\s*c$"), "O(1)"),
    (re.compile(r"^c\s*\*\s*k$"), "O(1)"),
]


def _tighten(expr: str) -> str:
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
    if not raw:
        return None
    s = pre_canonicalize(raw)
    if not s:
        return None
    inner = extract_inner(s)
    if "sqrt" in inner:
        return None
    if "?" in inner:
        return None
    inner = normalize_variables(inner)
    inner = _tighten(inner)
    for pat, lab in _PATTERNS:
        if pat.match(inner):
            assert lab in POINT_LABELS
            return lab
    return None


# Common raw labels from CodeComplex (codeparrot/codecomplex) mapped directly.
# These don't fit the O(...) regex machine because they're bare words.
_WORD_LABELS: dict[str, str] = {
    "constant": "O(1)",
    "linear": "O(n)",
    "quadratic": "O(n^2)",
    "cubic": "O(n^3)",
    "logn": "O(log n)",
    "log(n)": "O(log n)",
    "nlogn": "O(n log n)",
    "nlog(n)": "O(n log n)",
    "np": "exponential",
    "np-hard": "exponential",
    "exponential": "exponential",
    "factorial": "exponential",
}


def normalize_any(raw: str) -> str | None:
    """Normalize both O(...) strings and bare-word labels."""
    if not raw:
        return None
    key = raw.strip().lower().replace(" ", "")
    if key in _WORD_LABELS:
        return _WORD_LABELS[key]
    return normalize(raw)
