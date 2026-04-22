"""Vendored from microsoft/CodeBERT (MIT) — GraphCodeBERT/clonedetection/parser.

Upstream: https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT
Fetched for the Python DFG extractor `DFG_python`.
"""

from .DFG import DFG_python  # noqa: F401
from .utils import (  # noqa: F401
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
    tree_to_variable_index,
)
