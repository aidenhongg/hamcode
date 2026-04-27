"""Syntax-check every .py file in the project (excluding data/)."""
import ast
import sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = 0
    n_files = 0
    for p in sorted(root.glob("**/*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("data/") or rel.startswith("runs/"):
            continue
        n_files += 1
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            print(f"SYNTAX ERROR {rel}:{e.lineno}: {e.msg}")
            errors += 1
    print(f"{n_files} files checked, {errors} errors")
    return 0 if errors == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
