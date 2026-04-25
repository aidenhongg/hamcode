"""Throwaway validation: LongCoder tokenizer encodes each target language and
peft LoRA wraps the LongCoder classifier without errors. CPU-only smoke test.

Run from repo root: python codebert/tests/_validate_lora.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import LongCoderClassifier  # noqa: E402

SNIPPETS = {
    "python": "def two_sum(nums, t):\n    d = {}\n    for i, x in enumerate(nums):\n        if t-x in d: return [d[t-x], i]\n        d[x] = i\n",
    "java": "class Solution { int[] twoSum(int[] n, int t){ Map<Integer,Integer> d=new HashMap<>(); for(int i=0;;++i){int x=n[i],y=t-x; if(d.containsKey(y)) return new int[]{d.get(y),i}; d.put(x,i);} } }",
    "cpp": "vector<int> twoSum(vector<int>& n, int t){ unordered_map<int,int> d; for(int i=0;;++i){int x=n[i],y=t-x; if(d.count(y)) return {d[y],i}; d[x]=i;} }",
    "c": "#include <stdio.h>\nint* twoSum(int* n, int sz, int t){ static int r[2]; for(int i=0;i<sz;i++) for(int j=i+1;j<sz;j++) if(n[i]+n[j]==t){r[0]=i;r[1]=j;return r;} return 0; }",
    "csharp": "class Solution { public int[] TwoSum(int[] n, int t){ var d=new Dictionary<int,int>(); for(int i=0;;i++){int x=n[i],y=t-x; if(d.ContainsKey(y)) return new[]{d[y],i}; d[x]=i;} } }",
    "go": "package main\nimport \"fmt\"\nfunc twoSum(n []int, t int) []int { d:=map[int]int{}; for i,x:=range n { if j,ok:=d[t-x]; ok { return []int{j,i} }; d[x]=i }; return nil }",
    "javascript": "var twoSum = function(n, t){ const d = new Map(); for(let i=0;;++i){const x=n[i],y=t-x; if(d.has(y)) return [d.get(y),i]; d.set(x,i);} };",
    "typescript": "function twoSum(n: number[], t: number): number[] { const d = new Map<number,number>(); for(let i=0;;++i){const x=n[i],y=t-x; if(d.has(y)) return [d.get(y)!,i]; d.set(x,i);} }",
    "php": "<?php class Solution { function twoSum($n,$t){$d=[]; foreach($n as $i=>$x){$y=$t-$x; if(isset($d[$y])) return [$d[$y],$i]; $d[$x]=$i;}} } ?>",
    "ruby": "def two_sum(nums, t)\n  d = {}\n  nums.each_with_index { |x, i| return [d[t-x], i] if d.key?(t-x); d[x]=i }\nend",
    "rust": "fn two_sum(n: Vec<i32>, t: i32) -> Vec<i32> { let mut d=std::collections::HashMap::new(); for (i,&x) in n.iter().enumerate(){ if let Some(&j)=d.get(&(t-x)){return vec![j as i32,i as i32];} d.insert(x, i); } vec![] }",
    "swift": "class Solution { func twoSum(_ n: [Int], _ t: Int) -> [Int] { var d=[Int:Int](); for (i,x) in n.enumerated(){ if let j=d[t-x] { return [j,i] }; d[x]=i } ; return [] } }",
}


def main() -> int:
    print("=== Test 1: LongCoder tokenizer encodes all 11 target languages ===")
    tok = AutoTokenizer.from_pretrained("microsoft/longcoder-base")
    print(f"  vocab_size={tok.vocab_size}  cls={tok.cls_token_id}  sep={tok.sep_token_id}  unk={tok.unk_token_id}")
    print(f"  {'lang':12s} {'chars':>6s}  {'tokens':>6s}  {'cpr':>5s}  preview")
    for lang, src in SNIPPETS.items():
        ids = tok(src, add_special_tokens=False)["input_ids"]
        cpr = len(src) / max(1, len(ids))
        preview = " ".join(tok.convert_ids_to_tokens(ids[:8]))[:60]
        print(f"  {lang:12s} {len(src):6d}  {len(ids):6d}  {cpr:5.2f}  {preview}")

    print("\n=== Test 2: peft LoRA wraps LongCoderClassifier ===")
    from peft import LoraConfig, TaskType, get_peft_model

    print("  building LongCoderClassifier (cpu) ...")
    model = LongCoderClassifier(model_name="microsoft/longcoder-base")
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  base params: total={n_total:,}")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        # Longformer attention modules: query, key, value (local) + query_global, key_global, value_global
        target_modules=["query", "key", "value", "query_global", "key_global", "value_global"],
        modules_to_save=["classifier"],   # train the classifier head alongside the adapter
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    print(f"  applying LoraConfig r=16 alpha=32 ...")
    pmodel = get_peft_model(model, lora_cfg)
    n_total2 = sum(p.numel() for p in pmodel.parameters())
    n_train = sum(p.numel() for p in pmodel.parameters() if p.requires_grad)
    print(f"  after wrap: total={n_total2:,}  trainable={n_train:,} ({100*n_train/n_total:.2f}%)")

    print("\n=== Test 3: forward pass with bridge + memory bundle ===")
    # The forward signature on model.py absorbs **_unused, so peft's
    # inputs_embeds=None doesn't crash. No inline patch needed.
    pmodel.eval()
    seq = 64
    input_ids = torch.zeros((1, seq), dtype=torch.long); input_ids[0, 0] = tok.cls_token_id
    input_ids[0, -1] = tok.sep_token_id
    attn = torch.ones((1, seq), dtype=torch.long)
    g_attn = torch.zeros((1, seq), dtype=torch.long); g_attn[0, 0] = 1; g_attn[0, -1] = 1
    types = torch.zeros((1, seq), dtype=torch.long)
    with torch.no_grad():
        logits = pmodel(input_ids=input_ids, attention_mask=attn,
                        global_attention_mask=g_attn, token_type_ids=types)
    print(f"  forward ok: logits shape={tuple(logits.shape)}")

    print("\n=== Test 4: per-language LoRA save/load roundtrip ===")
    out = Path("runs/_lora_smoke")
    out.mkdir(parents=True, exist_ok=True)
    pmodel.save_pretrained(str(out))
    files = sorted(p.name for p in out.iterdir())
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    print(f"  saved files: {files}")
    print(f"  adapter size: {sizes}")
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
