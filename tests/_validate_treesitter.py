"""Throwaway validation: tree-sitter parses all 11 target languages.

Run from repo root: python codebert/tests/_validate_treesitter.py
"""

from __future__ import annotations

import importlib
import sys

from tree_sitter import Language, Parser


def _walk(node):
    yield node
    for c in node.children:
        yield from _walk(c)


SAMPLES = [
    ("python", "tree_sitter_python", "language", "def f(x):\n    return x + 1\n"),
    ("java", "tree_sitter_java", "language", "class A { int f(int x){return x+1;} }"),
    ("cpp", "tree_sitter_cpp", "language", "#include <iostream>\nint f(int x){return x+1;}"),
    ("c", "tree_sitter_c", "language", "#include <stdio.h>\nint f(int x){return x+1;}"),
    ("csharp", "tree_sitter_c_sharp", "language", "class A { int F(int x) => x + 1; }"),
    ("go", "tree_sitter_go", "language", "package main\nfunc f(x int) int { return x + 1 }"),
    ("javascript", "tree_sitter_javascript", "language", "function f(x){return x+1}"),
    ("typescript", "tree_sitter_typescript", "language_typescript",
     "function f(x: number): number { return x + 1; }"),
    ("php", "tree_sitter_php", "language_php", "<?php function f($x){return $x+1;} ?>"),
    ("ruby", "tree_sitter_ruby", "language", "def f(x); x+1; end"),
    ("rust", "tree_sitter_rust", "language", "fn f(x:i32) -> i32 { x + 1 }"),
    ("swift", "tree_sitter_swift", "language", "func f(x: Int) -> Int { return x + 1 }"),
]


def main() -> int:
    failures: list[tuple[str, str]] = []
    for name, mod_name, fn_name, src in SAMPLES:
        try:
            mod = importlib.import_module(mod_name)
            lang = Language(getattr(mod, fn_name)())
            parser = Parser(lang)
            tree = parser.parse(bytes(src, "utf-8"))
            root = tree.root_node
            has_err = root.has_error
            n_nodes = sum(1 for _ in _walk(root))
            print(f"{name:12s} {mod_name:24s} root={root.type:18s} "
                  f"nodes={n_nodes:4d} has_error={has_err}")
            if has_err:
                failures.append((name, "has_error"))
        except Exception as e:
            failures.append((name, type(e).__name__ + ": " + str(e)[:160]))
            print(f"{name:12s} FAIL: {e}")
    if failures:
        print("---")
        for f in failures:
            print("FAIL", f)
        return 1
    print("---\nAll 11 languages parse cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
