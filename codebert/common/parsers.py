"""Per-language tree-sitter parser registry + helpers.

One source of truth for: language identifiers, the PyPI module each
identifier maps to, the language-grammar function name (most are `language`,
but tree_sitter_typescript and tree_sitter_php expose multiple grammars), and
the per-language node-kind sets used downstream:

  - MEMORY_NODE_KINDS: nodes whose terminating newline gets a memory token
    (long-range global-attention slots; LongCoder's design).
  - LOOP_NODE_KINDS: nodes that count toward `nested_loop_depth` and similar
    AST-feature counters.
  - DECISION_NODE_KINDS: McCabe-style cyclomatic decision points.
  - FUNCTION_NODE_KINDS: function/method definition nodes.
  - CLASS_NODE_KINDS: class/struct definition nodes.
  - IMPORT_NODE_KINDS: import / include / use directives.
  - IF_NODE_KINDS, SWITCH_NODE_KINDS, BREAK_NODE_KINDS, RETURN_NODE_KINDS,
    JUMP_NODE_KINDS, TRY_NODE_KINDS, BOOL_OP_NODE_KINDS

All sets are derived from each grammar's published node-type vocabulary —
see tree-sitter-{lang}/src/node-types.json upstream. Adding a missing kind
is a one-line edit; missing kinds make the corresponding feature undercount
but never crash.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from tree_sitter import Language, Parser

from .schemas import LANGUAGES


# ---------------------------------------------------------------------------
# PyPI module + grammar-fn map
# ---------------------------------------------------------------------------

# (canonical_name -> (pypi_module, fn_name)).  fn_name is the attribute on the
# imported module that returns a `tree_sitter.Language` capsule.
_GRAMMAR_MAP: dict[str, tuple[str, str]] = {
    "python":     ("tree_sitter_python",     "language"),
    "java":       ("tree_sitter_java",       "language"),
    "cpp":        ("tree_sitter_cpp",        "language"),
    "c":          ("tree_sitter_c",          "language"),
    "csharp":     ("tree_sitter_c_sharp",    "language"),
    "go":         ("tree_sitter_go",         "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "php":        ("tree_sitter_php",        "language_php"),
    "ruby":       ("tree_sitter_ruby",       "language"),
    "rust":       ("tree_sitter_rust",       "language"),
    "swift":      ("tree_sitter_swift",      "language"),
}
assert set(_GRAMMAR_MAP) == set(LANGUAGES), "parser map drifted from LANGUAGES"


# ---------------------------------------------------------------------------
# Per-language node-kind sets
# ---------------------------------------------------------------------------

# Memory tokens are inserted at end-of-line for these kinds (statements that
# carry strong long-range signal: imports + function/class headers).
MEMORY_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({
        "import_statement", "import_from_statement", "future_import_statement",
        "function_definition", "class_definition", "decorated_definition",
    }),
    "java": frozenset({
        "import_declaration", "package_declaration",
        "class_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "method_declaration", "constructor_declaration",
    }),
    "cpp": frozenset({
        "preproc_include", "preproc_def", "using_declaration",
        "namespace_definition", "class_specifier", "struct_specifier",
        "function_definition", "template_declaration",
    }),
    "c": frozenset({
        "preproc_include", "preproc_def", "preproc_function_def",
        "function_definition", "struct_specifier", "union_specifier",
        "enum_specifier", "type_definition",
    }),
    "csharp": frozenset({
        "using_directive", "namespace_declaration", "file_scoped_namespace_declaration",
        "class_declaration", "interface_declaration", "struct_declaration",
        "enum_declaration", "record_declaration", "method_declaration",
        "constructor_declaration",
    }),
    "go": frozenset({
        "import_declaration", "import_spec_list", "package_clause",
        "function_declaration", "method_declaration",
        "type_declaration",
    }),
    "javascript": frozenset({
        "import_statement", "export_statement",
        "function_declaration", "class_declaration", "method_definition",
    }),
    "typescript": frozenset({
        "import_statement", "export_statement",
        "function_declaration", "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration", "enum_declaration",
    }),
    "php": frozenset({
        "namespace_definition", "namespace_use_declaration",
        "class_declaration", "interface_declaration", "trait_declaration",
        "function_definition", "method_declaration",
    }),
    "ruby": frozenset({
        "class", "module", "method", "singleton_method",
    }),
    "rust": frozenset({
        "use_declaration", "extern_crate_declaration", "mod_item",
        "function_item", "struct_item", "enum_item", "trait_item",
        "impl_item",
    }),
    "swift": frozenset({
        "import_declaration",
        "class_declaration", "protocol_declaration",
        "function_declaration", "init_declaration", "deinit_declaration",
        "subscript_declaration", "computed_property",
    }),
}

# `for` / `while` / `do` style loops.
LOOP_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({"for_statement", "while_statement"}),
    "java": frozenset({"for_statement", "enhanced_for_statement",
                       "while_statement", "do_statement"}),
    "cpp": frozenset({"for_statement", "for_range_loop",
                      "while_statement", "do_statement"}),
    "c": frozenset({"for_statement", "while_statement", "do_statement"}),
    "csharp": frozenset({"for_statement", "foreach_statement",
                         "while_statement", "do_statement"}),
    "go": frozenset({"for_statement"}),
    "javascript": frozenset({"for_statement", "for_in_statement",
                             "for_of_statement", "while_statement", "do_statement"}),
    "typescript": frozenset({"for_statement", "for_in_statement",
                             "for_of_statement", "while_statement", "do_statement"}),
    "php": frozenset({"for_statement", "foreach_statement",
                      "while_statement", "do_statement"}),
    "ruby": frozenset({"for", "while", "until", "do_block"}),
    "rust": frozenset({"for_expression", "while_expression", "loop_expression"}),
    "swift": frozenset({"for_statement", "while_statement", "repeat_while_statement"}),
}

# McCabe-style decision points (each adds 1 to cyclomatic complexity).
DECISION_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({
        "if_statement", "elif_clause", "for_statement", "while_statement",
        "except_clause", "match_statement", "case_clause",
        "boolean_operator",  # `and`, `or` — each adds a decision in McCabe ext.
        "conditional_expression",  # ternary
    }),
    "java": frozenset({
        "if_statement", "for_statement", "enhanced_for_statement",
        "while_statement", "do_statement", "switch_label",
        "catch_clause", "ternary_expression",
    }),
    "cpp": frozenset({
        "if_statement", "for_statement", "for_range_loop", "while_statement",
        "do_statement", "case_statement", "catch_clause",
        "conditional_expression",
    }),
    "c": frozenset({
        "if_statement", "for_statement", "while_statement", "do_statement",
        "case_statement", "conditional_expression",
    }),
    "csharp": frozenset({
        "if_statement", "for_statement", "foreach_statement",
        "while_statement", "do_statement", "switch_section",
        "catch_clause", "conditional_expression",
    }),
    "go": frozenset({
        "if_statement", "for_statement", "expression_case", "default_case",
        "type_case", "type_switch_statement",
    }),
    "javascript": frozenset({
        "if_statement", "for_statement", "for_in_statement", "for_of_statement",
        "while_statement", "do_statement", "switch_case",
        "catch_clause", "ternary_expression",
    }),
    "typescript": frozenset({
        "if_statement", "for_statement", "for_in_statement", "for_of_statement",
        "while_statement", "do_statement", "switch_case",
        "catch_clause", "ternary_expression",
    }),
    "php": frozenset({
        "if_statement", "for_statement", "foreach_statement", "while_statement",
        "do_statement", "case_statement", "catch_clause", "conditional_expression",
    }),
    "ruby": frozenset({
        "if", "elsif", "unless", "for", "while", "until",
        "when", "rescue", "conditional", "ternary",
    }),
    "rust": frozenset({
        "if_expression", "match_arm", "match_expression",
        "for_expression", "while_expression", "loop_expression",
    }),
    "swift": frozenset({
        "if_statement", "for_statement", "while_statement",
        "repeat_while_statement", "switch_entry", "guard_statement",
        "catch_clause",
    }),
}

# Function definition nodes (used to detect recursion + count methods).
FUNCTION_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition"}),
    "java": frozenset({"method_declaration", "constructor_declaration"}),
    "cpp": frozenset({"function_definition"}),
    "c": frozenset({"function_definition"}),
    "csharp": frozenset({"method_declaration", "constructor_declaration",
                         "local_function_statement"}),
    "go": frozenset({"function_declaration", "method_declaration",
                     "func_literal"}),
    "javascript": frozenset({"function_declaration", "function_expression",
                             "arrow_function", "method_definition"}),
    "typescript": frozenset({"function_declaration", "function_expression",
                             "arrow_function", "method_definition"}),
    "php": frozenset({"function_definition", "method_declaration"}),
    "ruby": frozenset({"method", "singleton_method"}),
    "rust": frozenset({"function_item", "closure_expression"}),
    "swift": frozenset({"function_declaration", "init_declaration"}),
}

CLASS_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({"class_definition"}),
    "java": frozenset({"class_declaration", "interface_declaration",
                       "enum_declaration", "record_declaration"}),
    "cpp": frozenset({"class_specifier", "struct_specifier"}),
    "c": frozenset({"struct_specifier", "union_specifier"}),
    "csharp": frozenset({"class_declaration", "interface_declaration",
                         "struct_declaration", "record_declaration",
                         "enum_declaration"}),
    "go": frozenset({"type_declaration", "struct_type", "interface_type"}),
    "javascript": frozenset({"class_declaration"}),
    "typescript": frozenset({"class_declaration", "interface_declaration"}),
    "php": frozenset({"class_declaration", "interface_declaration",
                      "trait_declaration"}),
    "ruby": frozenset({"class", "module"}),
    "rust": frozenset({"struct_item", "enum_item", "trait_item",
                       "impl_item", "type_item"}),
    "swift": frozenset({"class_declaration", "protocol_declaration"}),
}

IF_NODE_KINDS: dict[str, frozenset[str]] = {
    lang: frozenset(k for k in DECISION_NODE_KINDS[lang]
                    if k.startswith("if_") or k in {"elif_clause", "if", "elsif",
                                                    "unless", "if_expression"})
    for lang in LANGUAGES
}

SWITCH_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({"match_statement"}),
    "java": frozenset({"switch_expression", "switch_statement"}),
    "cpp": frozenset({"switch_statement"}),
    "c": frozenset({"switch_statement"}),
    "csharp": frozenset({"switch_statement", "switch_expression"}),
    "go": frozenset({"expression_switch_statement", "type_switch_statement"}),
    "javascript": frozenset({"switch_statement"}),
    "typescript": frozenset({"switch_statement"}),
    "php": frozenset({"switch_statement", "match_expression"}),
    "ruby": frozenset({"case", "case_match"}),
    "rust": frozenset({"match_expression"}),
    "swift": frozenset({"switch_statement"}),
}

BREAK_NODE_KINDS: dict[str, frozenset[str]] = {
    lang: frozenset({"break_statement"}) for lang in LANGUAGES
}
BREAK_NODE_KINDS["ruby"] = frozenset({"break"})
BREAK_NODE_KINDS["rust"] = frozenset({"break_expression"})

RETURN_NODE_KINDS: dict[str, frozenset[str]] = {
    lang: frozenset({"return_statement"}) for lang in LANGUAGES
}
RETURN_NODE_KINDS["ruby"] = frozenset({"return"})
RETURN_NODE_KINDS["rust"] = frozenset({"return_expression"})

JUMP_NODE_KINDS: dict[str, frozenset[str]] = {
    lang: BREAK_NODE_KINDS[lang] | RETURN_NODE_KINDS[lang] | frozenset({"continue_statement"})
    for lang in LANGUAGES
}
JUMP_NODE_KINDS["ruby"] = JUMP_NODE_KINDS["ruby"] | frozenset({"continue", "next"})
JUMP_NODE_KINDS["rust"] = JUMP_NODE_KINDS["rust"] | frozenset({"continue_expression"})
JUMP_NODE_KINDS["go"] = JUMP_NODE_KINDS["go"] | frozenset({"goto_statement", "fallthrough_statement"})

# Boolean operator nodes — each &&/||/and/or adds a McCabe decision point.
BOOL_OP_NODE_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({"boolean_operator"}),
    "java": frozenset({"binary_expression"}),  # filtered by op text downstream
    "cpp": frozenset({"binary_expression"}),
    "c": frozenset({"binary_expression"}),
    "csharp": frozenset({"binary_expression"}),
    "go": frozenset({"binary_expression"}),
    "javascript": frozenset({"binary_expression"}),
    "typescript": frozenset({"binary_expression"}),
    "php": frozenset({"binary_expression"}),
    "ruby": frozenset({"binary"}),
    "rust": frozenset({"binary_expression"}),
    "swift": frozenset({"prefix_expression"}),  # Swift uses op-wrapped expressions
}


# ---------------------------------------------------------------------------
# Parser cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_parser(language: str) -> Parser:
    """Return a tree-sitter Parser for the given canonical language name.

    Raises KeyError if the language isn't known to us; ImportError or RuntimeError
    if the corresponding tree-sitter PyPI package isn't installed.
    """
    if language not in _GRAMMAR_MAP:
        raise KeyError(f"unknown language: {language!r} "
                       f"(known: {sorted(_GRAMMAR_MAP)})")
    mod_name, fn_name = _GRAMMAR_MAP[language]
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ImportError(
            f"missing tree-sitter package for {language!r}: pip install {mod_name.replace('_', '-')}"
        ) from e
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"{mod_name} has no attribute {fn_name!r}")
    lang = Language(fn())
    return Parser(lang)


def parse(language: str, code: str) -> Any:
    """Parse `code` in `language` and return the tree-sitter Tree object."""
    return get_parser(language).parse(bytes(code, "utf-8"))


def syntax_ok(language: str, code: str) -> bool:
    """Return True iff parsing `code` produces an error-free tree."""
    try:
        tree = parse(language, code)
    except Exception:
        return False
    return not tree.root_node.has_error


def walk(node) -> "Iterator[Any]":
    """Pre-order iterator over a tree-sitter node tree."""
    yield node
    for child in node.children:
        yield from walk(child)


def memory_byte_offsets(language: str, code: str) -> list[int]:
    """Byte offsets of the `\\n` that ends each statement in MEMORY_NODE_KINDS.

    Falls back to an empty list on parse failure — the memory-token marking is
    a soft signal; LongCoder still trains without it.
    """
    try:
        tree = parse(language, code)
    except Exception:
        return []
    code_b = code.encode("utf-8")
    n = len(code_b)
    out: list[int] = []
    kinds = MEMORY_NODE_KINDS.get(language, frozenset())

    def visit(node) -> None:
        if node.type in kinds:
            end = node.end_byte
            if 0 < end <= n and code_b[end - 1:end] == b"\n":
                out.append(end - 1)
            else:
                nl = code_b.find(b"\n", end)
                if nl != -1:
                    out.append(nl)
            return
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Self-test (run: python -m common.parsers)
# ---------------------------------------------------------------------------

_SELF_TEST_SAMPLES: dict[str, str] = {
    "python":     "def f(x):\n    return x + 1\n",
    "java":       "class A { int f(int x){return x+1;} }",
    "cpp":        "int f(int x){return x+1;}",
    "c":          "int f(int x){return x+1;}",
    "csharp":     "class A { int F(int x) => x + 1; }",
    "go":         "package main\nfunc f(x int) int { return x + 1 }",
    "javascript": "function f(x){return x+1}",
    "typescript": "function f(x: number): number { return x + 1; }",
    "php":        "<?php function f($x){return $x+1;} ?>",
    "ruby":       "def f(x); x+1; end",
    "rust":       "fn f(x:i32) -> i32 { x + 1 }",
    "swift":      "func f(x: Int) -> Int { return x + 1 }",
}


def _self_test() -> None:
    fails: list[str] = []
    for lang, src in _SELF_TEST_SAMPLES.items():
        try:
            ok = syntax_ok(lang, src)
            mem = memory_byte_offsets(lang, src)
            if not ok:
                fails.append(f"{lang}: syntax_ok returned False")
        except Exception as e:
            fails.append(f"{lang}: {type(e).__name__}: {e}")
    if fails:
        for f in fails:
            print("FAIL", f)
        raise SystemExit(f"{len(fails)} parser self-test failures")
    print(f"parsers self-test OK ({len(_SELF_TEST_SAMPLES)} languages)")


if __name__ == "__main__":
    _self_test()
