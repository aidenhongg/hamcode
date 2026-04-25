"""One-stop smoke test that exercises every piece a Runpod 4090 will hit.

Runs in <30s on CPU. Skips: actual training, HF tokenizer download, network.

Tests (each must pass):
  T1. parser registry parses all 12 languages
  T2. memory_byte_offsets returns sane values for each language
  T3. AST features schema is stable across languages
  T4. AST features extract a non-zero vector for each language
  T5. schemas: PointRecord/PairRecord round-trip and pyarrow schema includes language
  T6. pipeline parsers compile (every numeric-prefixed pipeline file)
  T7. predict.py extension detection covers all 11 + python
  T8. configs/point.yaml and configs/lora.yaml parse
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

THIS = Path(__file__).resolve()
REPO = THIS.parents[1]
sys.path.insert(0, str(REPO))


def test_t1_parsers() -> None:
    from common.parsers import _SELF_TEST_SAMPLES, syntax_ok
    for lang, src in _SELF_TEST_SAMPLES.items():
        assert syntax_ok(lang, src), f"T1 fail: {lang}"


def test_t2_memory_offsets() -> None:
    from common.parsers import memory_byte_offsets
    samples = {
        "python": "import os\n\ndef f():\n    pass\n",
        "java":   "package x;\nimport java.util.*;\nclass A{}\n",
        "cpp":    "#include <iostream>\nint f(){return 0;}\n",
        "go":     "package main\nimport \"fmt\"\nfunc f(){}\n",
        "rust":   "use std::io;\nfn f(){}\n",
        "swift":  "import Foundation\nfunc f(){}\n",
    }
    for lang, src in samples.items():
        offs = memory_byte_offsets(lang, src)
        assert isinstance(offs, list)
        # At minimum each sample has one import + one fn -> 2 offsets
        assert len(offs) >= 1, f"T2 {lang}: got {offs}"


def test_t3_ast_schema_stable() -> None:
    from stacking.features.ast_features import N_FEATURES, FEATURE_NAMES, FEATURE_KIND
    assert N_FEATURES == 21, N_FEATURES
    assert len(FEATURE_NAMES) == 21
    assert FEATURE_KIND["nested_loop_depth"] == "count"
    assert FEATURE_KIND["recursion_present"] == "bool"
    assert FEATURE_KIND["cyclomatic_max"] == "cont"


def test_t4_ast_features_per_language() -> None:
    from stacking.features.ast_features import extract_features, N_FEATURES
    samples = {
        "python":     "def lcs(a, b):\n    m=len(a); n=len(b)\n    dp=[[0]*(n+1) for _ in range(m+1)]\n    for i in range(1,m+1):\n        for j in range(1,n+1):\n            if a[i-1]==b[j-1]: dp[i][j]=dp[i-1][j-1]+1\n            else: dp[i][j]=max(dp[i-1][j],dp[i][j-1])\n    return dp[m][n]\n",
        "java":       "class S { static int lcs(String a, String b){ int m=a.length(),n=b.length(); int[][] dp = new int[m+1][n+1]; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a.charAt(i-1)==b.charAt(j-1) ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j],dp[i][j-1]); return dp[m][n]; } }",
        "cpp":        "int lcs(const char* a, const char* b){ int m=0,n=0; while(a[m]) m++; while(b[n]) n++; int dp[100][100]={}; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a[i-1]==b[j-1] ? dp[i-1][j-1]+1 : (dp[i-1][j] > dp[i][j-1] ? dp[i-1][j] : dp[i][j-1]); return dp[m][n]; }",
        "c":          "int lcs(const char* a, int m, const char* b, int n){ int dp[100][100]={}; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a[i-1]==b[j-1] ? dp[i-1][j-1]+1 : (dp[i-1][j] > dp[i][j-1] ? dp[i-1][j] : dp[i][j-1]); return dp[m][n]; }",
        "csharp":     "class S { public static int Lcs(string a, string b){ int m=a.Length,n=b.Length; var dp = new int[m+1,n+1]; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i,j] = a[i-1]==b[j-1] ? dp[i-1,j-1]+1 : System.Math.Max(dp[i-1,j],dp[i,j-1]); return dp[m,n]; } }",
        "go":         "package main\nfunc Lcs(a, b string) int { m, n := len(a), len(b); dp := make([][]int, m+1); for i := range dp { dp[i] = make([]int, n+1) }; for i := 1; i <= m; i++ { for j := 1; j <= n; j++ { if a[i-1] == b[j-1] { dp[i][j] = dp[i-1][j-1] + 1 } else if dp[i-1][j] > dp[i][j-1] { dp[i][j] = dp[i-1][j] } else { dp[i][j] = dp[i][j-1] } } }; return dp[m][n] }",
        "javascript": "function lcs(a, b){ const m=a.length, n=b.length; const dp = Array.from({length:m+1}, () => new Array(n+1).fill(0)); for(let i=1;i<=m;i++) for(let j=1;j<=n;j++){ dp[i][j] = a[i-1]===b[j-1] ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]); } return dp[m][n]; }",
        "typescript": "function lcs(a: string, b: string): number { const m=a.length, n=b.length; const dp: number[][] = Array.from({length:m+1}, () => new Array(n+1).fill(0)); for(let i=1;i<=m;i++) for(let j=1;j<=n;j++){ dp[i][j] = a[i-1]===b[j-1] ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]); } return dp[m][n]; }",
        "php":        "<?php function lcs($a, $b){ $m=strlen($a); $n=strlen($b); $dp=array_fill(0, $m+1, array_fill(0, $n+1, 0)); for($i=1;$i<=$m;$i++) for($j=1;$j<=$n;$j++) $dp[$i][$j] = $a[$i-1]===$b[$j-1] ? $dp[$i-1][$j-1]+1 : max($dp[$i-1][$j], $dp[$i][$j-1]); return $dp[$m][$n]; } ?>",
        "ruby":       "def lcs(a, b)\n  m, n = a.length, b.length\n  dp = Array.new(m+1) { Array.new(n+1, 0) }\n  (1..m).each do |i|\n    (1..n).each do |j|\n      dp[i][j] = a[i-1] == b[j-1] ? dp[i-1][j-1] + 1 : [dp[i-1][j], dp[i][j-1]].max\n    end\n  end\n  dp[m][n]\nend\n",
        "rust":       "fn lcs(a: &str, b: &str) -> usize { let (m,n) = (a.len(), b.len()); let mut dp = vec![vec![0usize; n+1]; m+1]; for i in 1..=m { for j in 1..=n { dp[i][j] = if a.as_bytes()[i-1] == b.as_bytes()[j-1] { dp[i-1][j-1] + 1 } else { std::cmp::max(dp[i-1][j], dp[i][j-1]) }; } } dp[m][n] }",
        "swift":      "func lcs(_ a: String, _ b: String) -> Int { let ac = Array(a); let bc = Array(b); let m = ac.count; let n = bc.count; var dp = Array(repeating: Array(repeating: 0, count: n+1), count: m+1); for i in 1...m { for j in 1...n { dp[i][j] = ac[i-1] == bc[j-1] ? dp[i-1][j-1] + 1 : max(dp[i-1][j], dp[i][j-1]) } }; return dp[m][n] }",
    }
    for lang, src in samples.items():
        f = extract_features(src, lang)
        assert f.values.shape == (N_FEATURES,)
        # An LCS-like algorithm should have at least 1 loop and a recognizable nesting.
        loop = f.to_dict()["no_of_loop"]
        depth = f.to_dict()["nested_loop_depth"]
        assert loop >= 1, f"T4 {lang}: no loops detected"
        assert depth >= 1, f"T4 {lang}: depth=0"


def test_t5_schemas() -> None:
    from common.schemas import (
        POINT_SCHEMA, PAIR_SCHEMA, LANGUAGES, LANG_SET,
        PointRecord, PairRecord, RawRecord,
    )
    assert "language" in POINT_SCHEMA.names
    assert "language" in PAIR_SCHEMA.names
    assert len(LANGUAGES) == 12
    assert LANG_SET == frozenset(LANGUAGES)
    pr = PointRecord(id="x", source="test", language="python", problem_id="1",
                     solution_idx=0, code="def f():pass", code_sha256="a",
                     label="O(1)", raw_complexity="O(1)", tokens_bpe=5, ast_nodes=3)
    assert pr.language == "python"
    par = PairRecord(pair_id="p1", language="python", code_a="", code_b="",
                     label_a="O(1)", label_b="O(1)", ternary="same",
                     same_problem=True, tokens_combined=10)
    assert par.language == "python"
    rr = RawRecord(source="test", language="java", problem_id="1",
                    solution_idx=0, code="class A{}", raw_complexity="O(1)")
    assert rr.language == "java"


def test_t6_pipeline_compiles() -> None:
    import py_compile
    for f in (REPO / "pipeline").glob("*.py"):
        py_compile.compile(str(f), doraise=True)


def test_t7_predict_extension_map() -> None:
    # Just import + spot-check the map.
    import sys as _sys, importlib
    _sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("predict", REPO / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    EXT = mod.EXT_TO_LANG
    assert EXT[".py"] == "python"
    assert EXT[".cpp"] == "cpp"
    assert EXT[".rs"] == "rust"
    assert EXT[".swift"] == "swift"
    # All 12 languages must be a target somewhere
    targets = set(EXT.values())
    for lang in ("python", "java", "cpp", "c", "csharp", "go", "javascript",
                  "typescript", "php", "ruby", "rust", "swift"):
        assert lang in targets, f"T7: {lang} not in EXT_TO_LANG values"


def test_t8_configs_parse() -> None:
    import yaml
    pcfg = yaml.safe_load((REPO / "configs/point.yaml").read_text())
    lcfg = yaml.safe_load((REPO / "configs/lora.yaml").read_text())
    assert pcfg["model_name"] == "microsoft/longcoder-base"
    assert pcfg["max_seq_len"] == 1024
    assert pcfg["balanced_sampler"] is True
    assert lcfg["r"] == 16
    assert lcfg["lora_alpha"] == 32
    assert "query_global" in lcfg["target_modules"]


def main() -> int:
    tests = [
        ("T1 parsers all-language", test_t1_parsers),
        ("T2 memory offsets",       test_t2_memory_offsets),
        ("T3 AST schema stable",    test_t3_ast_schema_stable),
        ("T4 AST features per-lang", test_t4_ast_features_per_language),
        ("T5 schemas",              test_t5_schemas),
        ("T6 pipeline compiles",    test_t6_pipeline_compiles),
        ("T7 predict extensions",   test_t7_predict_extension_map),
        ("T8 configs parse",        test_t8_configs_parse),
    ]
    fails = []
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
        except AssertionError as e:
            fails.append((name, "AssertionError: " + (str(e) or "<no msg>")))
            print(f"  FAIL {name}: {e}")
        except Exception as e:
            fails.append((name, type(e).__name__ + ": " + str(e)[:200]))
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
    if fails:
        print(f"\n{len(fails)} test(s) failed")
        return 1
    print(f"\nall {len(tests)} smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
