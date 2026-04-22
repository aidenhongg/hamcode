"""Extract the 18 complexity-signal features from a Python snippet via tree-sitter.

Feature list (per user spec / CodeComplex paper):
    no_of_ifs, no_of_switches, no_of_loop, no_of_break, nested_loop_depth,
    noOfMethods, noOfVariables, noOfStatements, noOfJumps, recursion_present,
    priority_queue_present, hash_set_present, hash_map_present, no_of_sort,
    cond_in_loop_freq, loop_in_cond_freq, loop_in_loop_freq, cond_in_cond_freq.

Python adjustments to the Java-originated spec:
    - list/dict/set comprehensions and generator expressions count as loops
      via their `for_in_clause` children (each clause = one nesting level).
    - `match_statement` (Python 3.10+) counts as a switch.
    - Lambdas do NOT count as methods; only `function_definition`.

Implementation is a single DFS over the tree-sitter AST. O(n) in node count.
"""

from __future__ import annotations

from typing import Any

from tree_sitter_languages import get_parser

FEATURE_NAMES: tuple[str, ...] = (
    "no_of_ifs",
    "no_of_switches",
    "no_of_loop",
    "no_of_break",
    "nested_loop_depth",
    "noOfMethods",
    "noOfVariables",
    "noOfStatements",
    "noOfJumps",
    "recursion_present",
    "priority_queue_present",
    "hash_set_present",
    "hash_map_present",
    "no_of_sort",
    "cond_in_loop_freq",
    "loop_in_cond_freq",
    "loop_in_loop_freq",
    "cond_in_cond_freq",
)
NUM_FEATURES = len(FEATURE_NAMES)

_COMP_TYPES: frozenset[str] = frozenset({
    "list_comprehension", "dictionary_comprehension",
    "set_comprehension", "generator_expression",
})
# Loops for ancestor checks and depth tracking. Comprehensions count here because
# their body executes iteratively; `for_in_clause` is NOT in this set because
# its containing comprehension already owns the "is a loop" semantics (the first
# clause is absorbed into the comp; extras are accounted for inside the comp
# handler).
_LOOP_TYPES: frozenset[str] = frozenset({"for_statement", "while_statement"}) | _COMP_TYPES
_COND_TYPES: frozenset[str] = frozenset({"if_statement"})

_SORT_ATTRS: frozenset[str] = frozenset({"sort", "sorted"})
_HEAP_ATTRS: frozenset[str] = frozenset({
    "heappush", "heappop", "heappushpop", "heapify", "heapreplace",
    "nsmallest", "nlargest",
})
_PQ_SYMBOL_IMPORTS: frozenset[str] = frozenset({
    "heappush", "heappop", "heappushpop", "heapify", "heapreplace",
    "nsmallest", "nlargest", "PriorityQueue",
})
_SET_BUILTINS: frozenset[str] = frozenset({"set", "frozenset"})
_DICT_BUILTINS: frozenset[str] = frozenset({"dict", "defaultdict", "Counter", "OrderedDict"})

_STATEMENT_EXTRAS: frozenset[str] = frozenset({
    "function_definition", "class_definition", "decorated_definition",
})


_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        _parser = get_parser("python")
    return _parser


def _text(node) -> str:
    t = node.text
    if t is None:
        return ""
    return t.decode("utf-8", errors="replace")


def _collect_identifiers(node, out: set[str]) -> None:
    if node is None:
        return
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            out.add(_text(n))
            continue
        stack.extend(n.children)


def _param_name(param_node) -> str | None:
    """Extract the bound name from a parameter node (handles typed/default variants)."""
    if param_node is None:
        return None
    if param_node.type == "identifier":
        return _text(param_node)
    # typed_parameter / default_parameter / typed_default_parameter
    name = param_node.child_by_field_name("name")
    if name is not None and name.type == "identifier":
        return _text(name)
    # Fallback: first identifier child
    for c in param_node.children:
        if c.type == "identifier":
            return _text(c)
    return None


def parse(code: str):
    """Parse code and return (root_node, ast_node_count). Fails soft on binary input."""
    parser = _get_parser()
    tree = parser.parse(code.encode("utf-8", errors="replace"))
    root = tree.root_node
    count = 0
    stack = [root]
    while stack:
        n = stack.pop()
        count += 1
        stack.extend(n.children)
    return root, count


def extract_features(code: str) -> dict[str, int]:
    """Compute the 18 features. Returns a dict keyed by FEATURE_NAMES."""
    root, _ = parse(code)
    return _walk(root)


def extract_features_with_ast_count(code: str) -> tuple[dict[str, int], int]:
    """Compute the 18 features AND return the AST node count (used for filtering)."""
    root, n_nodes = parse(code)
    return _walk(root), n_nodes


def _walk(root) -> dict[str, int]:
    f: dict[str, int] = {name: 0 for name in FEATURE_NAMES}

    variables: set[str] = set()
    function_names: set[str] = set()
    call_names: list[str] = []
    imports: set[str] = set()
    import_symbols: set[str] = set()
    max_loop_depth = 0

    # Iterative DFS: stack of (node, ancestor_type_tuple, current_loop_depth).
    stack: list[tuple[Any, tuple[str, ...], int]] = [(root, (), 0)]

    while stack:
        node, anc, loop_depth = stack.pop()
        t = node.type

        # Statement counting
        if t.endswith("_statement") or t in _STATEMENT_EXTRAS:
            f["noOfStatements"] += 1

        custom_children_pushed = False

        # Conditionals
        if t == "if_statement":
            f["no_of_ifs"] += 1
            if any(a in _LOOP_TYPES for a in anc):
                f["cond_in_loop_freq"] += 1
            if "if_statement" in anc:
                f["cond_in_cond_freq"] += 1

        elif t == "match_statement":
            f["no_of_switches"] += 1

        # Statement loops
        elif t in ("for_statement", "while_statement"):
            f["no_of_loop"] += 1
            if any(a in _LOOP_TYPES for a in anc):
                f["loop_in_loop_freq"] += 1
            if "if_statement" in anc:
                f["loop_in_cond_freq"] += 1

        # Comprehensions: custom handling so nested comprehensions compute
        # nested_loop_depth correctly. Sibling `for_in_clause`s aren't ancestors
        # of each other in the tree, so we must account for them up front.
        elif t in _COMP_TYPES:
            if t == "set_comprehension":
                f["hash_set_present"] = 1
            elif t == "dictionary_comprehension":
                f["hash_map_present"] = 1

            for_clauses = sum(1 for c in node.children if c.type == "for_in_clause")
            total = max(1, for_clauses)
            f["no_of_loop"] += total
            has_outer_loop = any(a in _LOOP_TYPES for a in anc)
            if has_outer_loop:
                # All `total` loops are inside the outer loop
                f["loop_in_loop_freq"] += total
            # Extra clauses beyond the first are also nested inside clause 1
            f["loop_in_loop_freq"] += max(0, total - 1)
            if "if_statement" in anc:
                f["loop_in_cond_freq"] += total

            peak = loop_depth + total
            if peak > max_loop_depth:
                max_loop_depth = peak

            # Push children manually: body-like children at `peak` (conceptually
            # innermost), and `for_in_clause` / `if_clause` children at
            # loop_depth+1 (they execute at the comp's loop level). Skip the
            # default child-push.
            new_anc = anc + (t,)
            for child in reversed(node.children):
                if child.type in ("for_in_clause", "if_clause"):
                    stack.append((child, new_anc, loop_depth + 1))
                else:
                    stack.append((child, new_anc, peak))
            custom_children_pushed = True

        elif t == "for_in_clause":
            # Absorbed by parent comprehension. No counting here.
            pass

        elif t == "break_statement":
            f["no_of_break"] += 1
            f["noOfJumps"] += 1

        elif t in ("return_statement", "continue_statement", "raise_statement"):
            f["noOfJumps"] += 1

        elif t == "yield":
            f["noOfJumps"] += 1

        elif t == "function_definition":
            f["noOfMethods"] += 1
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                function_names.add(_text(name_node))
            params = node.child_by_field_name("parameters")
            if params is not None:
                for c in params.children:
                    pn = _param_name(c)
                    if pn:
                        variables.add(pn)

        elif t in ("assignment", "augmented_assignment"):
            lhs = node.child_by_field_name("left")
            _collect_identifiers(lhs, variables)

        elif t == "call":
            fn = node.child_by_field_name("function")
            if fn is not None:
                name = _text(fn)
                call_names.append(name)
                base = name.rsplit(".", 1)[-1]
                root_mod = name.split(".", 1)[0]

                if base in _SORT_ATTRS:
                    f["no_of_sort"] += 1

                if root_mod == "heapq" or base in _HEAP_ATTRS or base == "PriorityQueue":
                    f["priority_queue_present"] = 1

                if name in _SET_BUILTINS or base in _SET_BUILTINS:
                    f["hash_set_present"] = 1
                if name in _DICT_BUILTINS or base in _DICT_BUILTINS:
                    f["hash_map_present"] = 1

        elif t == "set":
            f["hash_set_present"] = 1

        elif t == "dictionary":
            f["hash_map_present"] = 1

        elif t == "import_statement":
            for c in node.children:
                if c.type == "dotted_name":
                    imports.add(_text(c).split(".", 1)[0])
                elif c.type == "aliased_import":
                    nm = c.child_by_field_name("name")
                    if nm is not None:
                        imports.add(_text(nm).split(".", 1)[0])

        elif t == "import_from_statement":
            mod = node.child_by_field_name("module_name")
            if mod is not None:
                imports.add(_text(mod).split(".", 1)[0])
            # Imported symbols: every dotted_name after the first (module) one
            seen_module = False
            for c in node.children:
                if c.type == "dotted_name":
                    if not seen_module:
                        seen_module = True
                    else:
                        import_symbols.add(_text(c))
                elif c.type == "aliased_import":
                    nm = c.child_by_field_name("name")
                    if nm is not None:
                        import_symbols.add(_text(nm))

        # Default child push (skipped for comprehensions, which handle their own)
        if not custom_children_pushed:
            # Track loop depth for nested_loop_depth (only statement loops here;
            # comprehensions bump max_loop_depth in their custom handler above).
            new_loop_depth = (
                loop_depth + 1 if t in ("for_statement", "while_statement") else loop_depth
            )
            if new_loop_depth > max_loop_depth:
                max_loop_depth = new_loop_depth

            new_anc = anc + (t,)
            for child in reversed(node.children):
                stack.append((child, new_anc, new_loop_depth))

    f["nested_loop_depth"] = max_loop_depth
    f["noOfVariables"] = len(variables)

    # Recursion: any call to a defined-function name
    for cn in call_names:
        base = cn.rsplit(".", 1)[-1]
        if base in function_names:
            f["recursion_present"] = 1
            break

    # Priority queue via imports
    if "heapq" in imports or (_PQ_SYMBOL_IMPORTS & import_symbols):
        f["priority_queue_present"] = 1

    return f
