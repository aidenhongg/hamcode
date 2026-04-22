"""Handcrafted-feature random forest for code time-complexity classification.

Point task only: given a single Python snippet, predict one of 11 complexity
classes defined in `complexity.labels`. Features computed from a tree-sitter
Python AST (see `complexity.features`).
"""
