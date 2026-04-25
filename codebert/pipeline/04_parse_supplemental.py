"""Generate supplemental synthetic samples for rare complexity classes,
in every language we train on.

Hand-crafted templates per (language, label) that cover the multi-variable
and tail classes (`O(m+n)`, `O(m*n)`, `O(m log n)`, `O((m+n) log(m+n))`,
`O(n^3)`, `exponential`). Each template is expanded via deterministic
variable renaming + harmless literal tweaks to produce K variants per
template. Records are tagged `source=synthetic` so we can ablation-test.

Augmentation only emits a variant if the renamed snippet still passes the
per-language tree-sitter syntax check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from common.parsers import syntax_ok
from common.schemas import LANGUAGES


# -----------------------------------------------------------------------------
# Per-language templates. {language: {label: [src, ...]}}
# Each src is a string with normal indentation; we don't dedent. Templates are
# intentionally short and self-contained so the tree-sitter check passes.
# -----------------------------------------------------------------------------

# Python — preserves the original 04_parse_supplemental tail templates.
PY_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''def subset_sum(arr, target):
    def rec(i, remain):
        if remain == 0:
            return True
        if i == len(arr) or remain < 0:
            return False
        return rec(i+1, remain-arr[i]) or rec(i+1, remain)
    return rec(0, target)
''',
        '''def permute(nums):
    out = []
    def back(path, rem):
        if not rem:
            out.append(path[:])
            return
        for i, x in enumerate(rem):
            path.append(x); back(path, rem[:i] + rem[i+1:]); path.pop()
    back([], list(nums))
    return out
''',
        '''def n_queens(n):
    res = []
    def back(row, cols, d1, d2, st):
        if row == n: res.append(st[:]); return
        for c in range(n):
            if c in cols or (row-c) in d1 or (row+c) in d2: continue
            st.append(c); cols.add(c); d1.add(row-c); d2.add(row+c)
            back(row+1, cols, d1, d2, st)
            st.pop(); cols.remove(c); d1.remove(row-c); d2.remove(row+c)
    back(0, set(), set(), set(), [])
    return res
''',
    ],
    "O(n^3)": [
        '''def floyd_warshall(dist, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
''',
        '''def matmul(A, B, n):
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
''',
    ],
    "O(m+n)": [
        '''def merge_sorted(a, b):
    out = []; i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]: out.append(a[i]); i += 1
        else: out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
''',
        '''from collections import deque
def bfs(adj, source, n):
    dist = [-1]*n; dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u]+1; q.append(v)
    return dist
''',
    ],
    "O(m*n)": [
        '''def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
''',
        '''def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
''',
    ],
    "O(m log n)": [
        '''import bisect
def search_many(queries, sorted_arr):
    return [bisect.bisect_left(sorted_arr, q) for q in queries]
''',
        '''import heapq
def process(events, n_initial):
    h = list(range(n_initial)); heapq.heapify(h); out = []
    for e in events:
        heapq.heappush(h, e)
        out.append(heapq.heappop(h))
    return out
''',
    ],
    "O((m+n) log(m+n))": [
        '''def combined_sort(a, b):
    merged = list(a) + list(b); merged.sort()
    return merged
''',
        '''def schedule(a, b):
    events = []
    for t in a: events.append((t, "a"))
    for t in b: events.append((t, "b"))
    events.sort()
    return events
''',
    ],
}


JAVA_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''class S { static boolean rec(int[] a, int i, int r){ if(r==0) return true; if(i==a.length||r<0) return false; return rec(a,i+1,r-a[i]) || rec(a,i+1,r); } static boolean subsetSum(int[] a, int t){ return rec(a,0,t); } }''',
        '''import java.util.*; class S { static List<List<Integer>> permute(int[] nums){ List<List<Integer>> res=new ArrayList<>(); back(res,new ArrayList<>(),nums,new boolean[nums.length]); return res; } static void back(List<List<Integer>> r, List<Integer> p, int[] n, boolean[] u){ if(p.size()==n.length){ r.add(new ArrayList<>(p)); return; } for(int i=0;i<n.length;i++){ if(u[i]) continue; u[i]=true; p.add(n[i]); back(r,p,n,u); p.remove(p.size()-1); u[i]=false; } } }''',
    ],
    "O(n^3)": [
        '''class S { static int[][] floydWarshall(int[][] dist, int n){ for(int k=0;k<n;k++) for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(dist[i][k]+dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k]+dist[k][j]; return dist; } }''',
        '''class S { static int[][] matmul(int[][] A, int[][] B, int n){ int[][] C = new int[n][n]; for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) C[i][j] += A[i][k]*B[k][j]; return C; } }''',
    ],
    "O(m+n)": [
        '''import java.util.*; class S { static int[] merge(int[] a, int[] b){ int[] out = new int[a.length+b.length]; int i=0,j=0,k=0; while(i<a.length && j<b.length) out[k++] = a[i] <= b[j] ? a[i++] : b[j++]; while(i<a.length) out[k++] = a[i++]; while(j<b.length) out[k++] = b[j++]; return out; } }''',
        '''import java.util.*; class S { static int[] bfs(List<List<Integer>> adj, int src, int n){ int[] dist = new int[n]; Arrays.fill(dist,-1); dist[src]=0; Deque<Integer> q = new ArrayDeque<>(); q.add(src); while(!q.isEmpty()){ int u=q.poll(); for(int v: adj.get(u)) if(dist[v]==-1){ dist[v]=dist[u]+1; q.add(v); } } return dist; } }''',
    ],
    "O(m*n)": [
        '''class S { static int lcs(String a, String b){ int m=a.length(), n=b.length(); int[][] dp = new int[m+1][n+1]; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a.charAt(i-1)==b.charAt(j-1) ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]); return dp[m][n]; } }''',
        '''class S { static int edit(String a, String b){ int m=a.length(), n=b.length(); int[][] dp=new int[m+1][n+1]; for(int i=0;i<=m;i++) dp[i][0]=i; for(int j=0;j<=n;j++) dp[0][j]=j; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a.charAt(i-1)==b.charAt(j-1) ? dp[i-1][j-1] : 1+Math.min(dp[i-1][j], Math.min(dp[i][j-1], dp[i-1][j-1])); return dp[m][n]; } }''',
    ],
    "O(m log n)": [
        '''import java.util.*; class S { static int[] searchMany(int[] queries, int[] sortedArr){ int[] r = new int[queries.length]; for(int i=0;i<queries.length;i++) r[i] = Arrays.binarySearch(sortedArr, queries[i]); return r; } }''',
        '''import java.util.*; class S { static int[] process(int[] events, int n){ PriorityQueue<Integer> h = new PriorityQueue<>(); for(int i=0;i<n;i++) h.add(i); int[] out = new int[events.length]; for(int i=0;i<events.length;i++){ h.add(events[i]); out[i] = h.poll(); } return out; } }''',
    ],
    "O((m+n) log(m+n))": [
        '''import java.util.*; class S { static int[] combinedSort(int[] a, int[] b){ int[] merged = new int[a.length+b.length]; System.arraycopy(a,0,merged,0,a.length); System.arraycopy(b,0,merged,a.length,b.length); Arrays.sort(merged); return merged; } }''',
    ],
}


CPP_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''#include <vector>
bool rec(const std::vector<int>& a, int i, int r){ if(r==0) return true; if(i==(int)a.size()||r<0) return false; return rec(a,i+1,r-a[i]) || rec(a,i+1,r); }
bool subset_sum(const std::vector<int>& a, int t){ return rec(a,0,t); }
''',
        '''#include <vector>
void back(std::vector<std::vector<int>>& r, std::vector<int>& p, std::vector<int>& nums){ if(p.size()==nums.size()){ r.push_back(p); return; } for(size_t i=0;i<nums.size();i++){ if(nums[i]==INT_MIN) continue; int x=nums[i]; nums[i]=INT_MIN; p.push_back(x); back(r,p,nums); p.pop_back(); nums[i]=x; } }
''',
    ],
    "O(n^3)": [
        '''#include <vector>
std::vector<std::vector<int>> floyd_warshall(std::vector<std::vector<int>> dist, int n){ for(int k=0;k<n;k++) for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(dist[i][k]+dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k]+dist[k][j]; return dist; }
''',
        '''#include <vector>
std::vector<std::vector<int>> matmul(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, int n){ std::vector<std::vector<int>> C(n, std::vector<int>(n,0)); for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) C[i][j] += A[i][k]*B[k][j]; return C; }
''',
    ],
    "O(m+n)": [
        '''#include <vector>
std::vector<int> merge_sorted(const std::vector<int>& a, const std::vector<int>& b){ std::vector<int> out; size_t i=0,j=0; while(i<a.size() && j<b.size()) out.push_back(a[i]<=b[j] ? a[i++] : b[j++]); while(i<a.size()) out.push_back(a[i++]); while(j<b.size()) out.push_back(b[j++]); return out; }
''',
    ],
    "O(m*n)": [
        '''#include <vector>
#include <string>
int lcs(const std::string& a, const std::string& b){ int m=a.size(), n=b.size(); std::vector<std::vector<int>> dp(m+1, std::vector<int>(n+1,0)); for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a[i-1]==b[j-1] ? dp[i-1][j-1]+1 : std::max(dp[i-1][j], dp[i][j-1]); return dp[m][n]; }
''',
    ],
    "O(m log n)": [
        '''#include <vector>
#include <algorithm>
std::vector<int> search_many(const std::vector<int>& queries, const std::vector<int>& sorted_arr){ std::vector<int> r; r.reserve(queries.size()); for(int q : queries) r.push_back((int)(std::lower_bound(sorted_arr.begin(), sorted_arr.end(), q) - sorted_arr.begin())); return r; }
''',
    ],
    "O((m+n) log(m+n))": [
        '''#include <vector>
#include <algorithm>
std::vector<int> combined_sort(const std::vector<int>& a, const std::vector<int>& b){ std::vector<int> merged = a; merged.insert(merged.end(), b.begin(), b.end()); std::sort(merged.begin(), merged.end()); return merged; }
''',
    ],
}


C_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''#include <stddef.h>
int rec(int* a, int n, int i, int r){ if(r==0) return 1; if(i==n||r<0) return 0; return rec(a,n,i+1,r-a[i]) || rec(a,n,i+1,r); }
int subset_sum(int* a, int n, int t){ return rec(a,n,0,t); }
''',
    ],
    "O(n^3)": [
        '''void floyd_warshall(int dist[][100], int n){ for(int k=0;k<n;k++) for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(dist[i][k]+dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k]+dist[k][j]; }
''',
        '''void matmul(int A[][100], int B[][100], int C[][100], int n){ for(int i=0;i<n;i++) for(int j=0;j<n;j++){ C[i][j]=0; for(int k=0;k<n;k++) C[i][j] += A[i][k]*B[k][j]; } }
''',
    ],
    "O(m+n)": [
        '''#include <stdlib.h>
int* merge_sorted(int* a, int m, int* b, int n, int* out_n){ int* out=(int*)malloc(sizeof(int)*(m+n)); int i=0,j=0,k=0; while(i<m && j<n) out[k++] = a[i]<=b[j] ? a[i++] : b[j++]; while(i<m) out[k++]=a[i++]; while(j<n) out[k++]=b[j++]; *out_n = m+n; return out; }
''',
    ],
    "O(m*n)": [
        '''int lcs(const char* a, int m, const char* b, int n, int dp[][100]){ for(int i=0;i<=m;i++) for(int j=0;j<=n;j++) dp[i][j]=0; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i][j] = a[i-1]==b[j-1] ? dp[i-1][j-1]+1 : (dp[i-1][j] > dp[i][j-1] ? dp[i-1][j] : dp[i][j-1]); return dp[m][n]; }
''',
    ],
    "O(m log n)": [
        '''int bsearch_one(int* arr, int n, int q){ int lo=0, hi=n; while(lo < hi){ int mid = (lo+hi)/2; if(arr[mid] < q) lo = mid+1; else hi = mid; } return lo; }
void search_many(int* arr, int n, int* queries, int m, int* out){ for(int i=0;i<m;i++) out[i] = bsearch_one(arr, n, queries[i]); }
''',
    ],
    "O((m+n) log(m+n))": [
        '''#include <stdlib.h>
int cmp_int(const void* x, const void* y){ return *(const int*)x - *(const int*)y; }
void combined_sort(int* a, int m, int* b, int n, int* out){ for(int i=0;i<m;i++) out[i]=a[i]; for(int j=0;j<n;j++) out[m+j]=b[j]; qsort(out, m+n, sizeof(int), cmp_int); }
''',
    ],
}


CSHARP_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''using System;
class S { static bool Rec(int[] a, int i, int r){ if(r==0) return true; if(i==a.Length||r<0) return false; return Rec(a,i+1,r-a[i]) || Rec(a,i+1,r); } public static bool SubsetSum(int[] a, int t) => Rec(a,0,t); }''',
    ],
    "O(n^3)": [
        '''class S { public static int[,] Floyd(int[,] dist, int n){ for(int k=0;k<n;k++) for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(dist[i,k]+dist[k,j] < dist[i,j]) dist[i,j] = dist[i,k]+dist[k,j]; return dist; } }''',
        '''class S { public static int[,] Matmul(int[,] A, int[,] B, int n){ var C = new int[n,n]; for(int i=0;i<n;i++) for(int j=0;j<n;j++) for(int k=0;k<n;k++) C[i,j] += A[i,k]*B[k,j]; return C; } }''',
    ],
    "O(m+n)": [
        '''using System.Collections.Generic;
class S { public static List<int> Merge(int[] a, int[] b){ var o = new List<int>(); int i=0,j=0; while(i<a.Length && j<b.Length){ if(a[i]<=b[j]){ o.Add(a[i]); i++; } else { o.Add(b[j]); j++; } } while(i<a.Length){ o.Add(a[i]); i++; } while(j<b.Length){ o.Add(b[j]); j++; } return o; } }''',
    ],
    "O(m*n)": [
        '''class S { public static int Lcs(string a, string b){ int m=a.Length, n=b.Length; var dp = new int[m+1, n+1]; for(int i=1;i<=m;i++) for(int j=1;j<=n;j++) dp[i,j] = a[i-1]==b[j-1] ? dp[i-1,j-1]+1 : System.Math.Max(dp[i-1,j], dp[i,j-1]); return dp[m,n]; } }''',
    ],
    "O(m log n)": [
        '''using System;
class S { public static int[] SearchMany(int[] q, int[] s){ var r = new int[q.Length]; for(int i=0;i<q.Length;i++) r[i] = Array.BinarySearch(s, q[i]); return r; } }''',
    ],
    "O((m+n) log(m+n))": [
        '''using System;
class S { public static int[] CombinedSort(int[] a, int[] b){ var m = new int[a.Length+b.Length]; Array.Copy(a, m, a.Length); Array.Copy(b, 0, m, a.Length, b.Length); Array.Sort(m); return m; } }''',
    ],
}


GO_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''package main
func rec(a []int, i, r int) bool { if r == 0 { return true }; if i == len(a) || r < 0 { return false }; return rec(a, i+1, r-a[i]) || rec(a, i+1, r) }
func SubsetSum(a []int, t int) bool { return rec(a, 0, t) }
''',
    ],
    "O(n^3)": [
        '''package main
func Floyd(dist [][]int, n int) [][]int { for k := 0; k < n; k++ { for i := 0; i < n; i++ { for j := 0; j < n; j++ { if dist[i][k]+dist[k][j] < dist[i][j] { dist[i][j] = dist[i][k] + dist[k][j] } } } }; return dist }
''',
        '''package main
func Matmul(A, B [][]int, n int) [][]int { C := make([][]int, n); for i := range C { C[i] = make([]int, n) }; for i := 0; i < n; i++ { for j := 0; j < n; j++ { for k := 0; k < n; k++ { C[i][j] += A[i][k] * B[k][j] } } }; return C }
''',
    ],
    "O(m+n)": [
        '''package main
func Merge(a, b []int) []int { out := make([]int, 0, len(a)+len(b)); i, j := 0, 0; for i < len(a) && j < len(b) { if a[i] <= b[j] { out = append(out, a[i]); i++ } else { out = append(out, b[j]); j++ } }; out = append(out, a[i:]...); out = append(out, b[j:]...); return out }
''',
    ],
    "O(m*n)": [
        '''package main
func Lcs(a, b string) int { m, n := len(a), len(b); dp := make([][]int, m+1); for i := range dp { dp[i] = make([]int, n+1) }; for i := 1; i <= m; i++ { for j := 1; j <= n; j++ { if a[i-1] == b[j-1] { dp[i][j] = dp[i-1][j-1] + 1 } else if dp[i-1][j] > dp[i][j-1] { dp[i][j] = dp[i-1][j] } else { dp[i][j] = dp[i][j-1] } } }; return dp[m][n] }
''',
    ],
    "O(m log n)": [
        '''package main
import "sort"
func SearchMany(q, s []int) []int { r := make([]int, len(q)); for i, x := range q { r[i] = sort.SearchInts(s, x) }; return r }
''',
    ],
    "O((m+n) log(m+n))": [
        '''package main
import "sort"
func CombinedSort(a, b []int) []int { m := append([]int{}, a...); m = append(m, b...); sort.Ints(m); return m }
''',
    ],
}


JS_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''function subsetSum(arr, target){ function rec(i, r){ if(r === 0) return true; if(i === arr.length || r < 0) return false; return rec(i+1, r-arr[i]) || rec(i+1, r); } return rec(0, target); }
''',
    ],
    "O(n^3)": [
        '''function floyd(dist, n){ for(let k=0;k<n;k++) for(let i=0;i<n;i++) for(let j=0;j<n;j++) if(dist[i][k]+dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k]+dist[k][j]; return dist; }
''',
        '''function matmul(A, B, n){ const C = Array.from({length:n}, () => new Array(n).fill(0)); for(let i=0;i<n;i++) for(let j=0;j<n;j++) for(let k=0;k<n;k++) C[i][j] += A[i][k]*B[k][j]; return C; }
''',
    ],
    "O(m+n)": [
        '''function mergeSorted(a, b){ const out = []; let i=0, j=0; while(i<a.length && j<b.length){ if(a[i] <= b[j]){ out.push(a[i++]); } else { out.push(b[j++]); } } while(i<a.length) out.push(a[i++]); while(j<b.length) out.push(b[j++]); return out; }
''',
    ],
    "O(m*n)": [
        '''function lcs(a, b){ const m=a.length, n=b.length; const dp = Array.from({length:m+1}, () => new Array(n+1).fill(0)); for(let i=1;i<=m;i++) for(let j=1;j<=n;j++){ dp[i][j] = a[i-1]===b[j-1] ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]); } return dp[m][n]; }
''',
    ],
    "O(m log n)": [
        '''function lowerBound(s, q){ let lo=0, hi=s.length; while(lo<hi){ const mid=(lo+hi)>>1; if(s[mid]<q) lo=mid+1; else hi=mid; } return lo; }
function searchMany(q, s){ return q.map(x => lowerBound(s, x)); }
''',
    ],
    "O((m+n) log(m+n))": [
        '''function combinedSort(a, b){ const merged = [...a, ...b]; merged.sort((x,y) => x-y); return merged; }
''',
    ],
}


TS_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''function subsetSum(arr: number[], target: number): boolean { function rec(i: number, r: number): boolean { if(r === 0) return true; if(i === arr.length || r < 0) return false; return rec(i+1, r-arr[i]) || rec(i+1, r); } return rec(0, target); }
''',
    ],
    "O(n^3)": [
        '''function floyd(dist: number[][], n: number): number[][] { for(let k=0;k<n;k++) for(let i=0;i<n;i++) for(let j=0;j<n;j++) if(dist[i][k]+dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k]+dist[k][j]; return dist; }
''',
    ],
    "O(m+n)": [
        '''function mergeSorted(a: number[], b: number[]): number[] { const out: number[] = []; let i=0, j=0; while(i<a.length && j<b.length){ if(a[i] <= b[j]){ out.push(a[i++]); } else { out.push(b[j++]); } } while(i<a.length) out.push(a[i++]); while(j<b.length) out.push(b[j++]); return out; }
''',
    ],
    "O(m*n)": [
        '''function lcs(a: string, b: string): number { const m=a.length, n=b.length; const dp: number[][] = Array.from({length:m+1}, () => new Array(n+1).fill(0)); for(let i=1;i<=m;i++) for(let j=1;j<=n;j++){ dp[i][j] = a[i-1]===b[j-1] ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]); } return dp[m][n]; }
''',
    ],
    "O(m log n)": [
        '''function lowerBound(s: number[], q: number): number { let lo=0, hi=s.length; while(lo<hi){ const mid=(lo+hi)>>1; if(s[mid]<q) lo=mid+1; else hi=mid; } return lo; }
function searchMany(q: number[], s: number[]): number[] { return q.map(x => lowerBound(s, x)); }
''',
    ],
    "O((m+n) log(m+n))": [
        '''function combinedSort(a: number[], b: number[]): number[] { const merged: number[] = [...a, ...b]; merged.sort((x,y) => x-y); return merged; }
''',
    ],
}


PHP_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''<?php
function rec($a, $i, $r){ if($r === 0) return true; if($i === count($a) || $r < 0) return false; return rec($a, $i+1, $r-$a[$i]) || rec($a, $i+1, $r); }
function subset_sum($a, $t){ return rec($a, 0, $t); }
?>''',
    ],
    "O(n^3)": [
        '''<?php
function floyd($dist, $n){ for($k=0; $k<$n; $k++) for($i=0; $i<$n; $i++) for($j=0; $j<$n; $j++) if($dist[$i][$k]+$dist[$k][$j] < $dist[$i][$j]) $dist[$i][$j] = $dist[$i][$k]+$dist[$k][$j]; return $dist; }
?>''',
    ],
    "O(m+n)": [
        '''<?php
function merge_sorted($a, $b){ $out=[]; $i=0; $j=0; while($i<count($a) && $j<count($b)){ if($a[$i] <= $b[$j]){ $out[]=$a[$i++]; } else { $out[]=$b[$j++]; } } while($i<count($a)) $out[]=$a[$i++]; while($j<count($b)) $out[]=$b[$j++]; return $out; }
?>''',
    ],
    "O(m*n)": [
        '''<?php
function lcs($a, $b){ $m=strlen($a); $n=strlen($b); $dp=array_fill(0, $m+1, array_fill(0, $n+1, 0)); for($i=1;$i<=$m;$i++) for($j=1;$j<=$n;$j++) $dp[$i][$j] = $a[$i-1]===$b[$j-1] ? $dp[$i-1][$j-1]+1 : max($dp[$i-1][$j], $dp[$i][$j-1]); return $dp[$m][$n]; }
?>''',
    ],
    "O(m log n)": [
        '''<?php
function lower_bound($s, $q){ $lo=0; $hi=count($s); while($lo<$hi){ $mid=intdiv($lo+$hi,2); if($s[$mid]<$q) $lo=$mid+1; else $hi=$mid; } return $lo; }
function search_many($q, $s){ $r=[]; foreach($q as $x) $r[]=lower_bound($s, $x); return $r; }
?>''',
    ],
    "O((m+n) log(m+n))": [
        '''<?php
function combined_sort($a, $b){ $m = array_merge($a, $b); sort($m); return $m; }
?>''',
    ],
}


RUBY_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''def rec(a, i, r)
  return true if r == 0
  return false if i == a.length || r < 0
  rec(a, i+1, r-a[i]) || rec(a, i+1, r)
end
def subset_sum(a, t); rec(a, 0, t); end
''',
    ],
    "O(n^3)": [
        '''def floyd(dist, n)
  (0...n).each { |k| (0...n).each { |i| (0...n).each { |j| dist[i][j] = dist[i][k]+dist[k][j] if dist[i][k]+dist[k][j] < dist[i][j] } } }
  dist
end
''',
    ],
    "O(m+n)": [
        '''def merge_sorted(a, b)
  out = []; i = 0; j = 0
  while i < a.length && j < b.length
    if a[i] <= b[j]; out << a[i]; i += 1; else; out << b[j]; j += 1; end
  end
  out += a[i..]; out += b[j..]; out
end
''',
    ],
    "O(m*n)": [
        '''def lcs(a, b)
  m, n = a.length, b.length
  dp = Array.new(m+1) { Array.new(n+1, 0) }
  (1..m).each do |i|
    (1..n).each do |j|
      dp[i][j] = a[i-1] == b[j-1] ? dp[i-1][j-1] + 1 : [dp[i-1][j], dp[i][j-1]].max
    end
  end
  dp[m][n]
end
''',
    ],
    "O(m log n)": [
        '''def lower_bound(s, q)
  lo = 0; hi = s.length
  while lo < hi
    mid = (lo+hi) / 2
    if s[mid] < q; lo = mid+1; else; hi = mid; end
  end
  lo
end
def search_many(q, s); q.map { |x| lower_bound(s, x) }; end
''',
    ],
    "O((m+n) log(m+n))": [
        '''def combined_sort(a, b); (a + b).sort; end
''',
    ],
}


RUST_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''fn rec(a: &[i32], i: usize, r: i32) -> bool { if r == 0 { return true; } if i == a.len() || r < 0 { return false; } rec(a, i+1, r - a[i]) || rec(a, i+1, r) }
fn subset_sum(a: &[i32], t: i32) -> bool { rec(a, 0, t) }
''',
    ],
    "O(n^3)": [
        '''fn floyd(dist: &mut Vec<Vec<i32>>, n: usize) { for k in 0..n { for i in 0..n { for j in 0..n { if dist[i][k] + dist[k][j] < dist[i][j] { dist[i][j] = dist[i][k] + dist[k][j]; } } } } }
''',
        '''fn matmul(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>, n: usize) -> Vec<Vec<i32>> { let mut c = vec![vec![0; n]; n]; for i in 0..n { for j in 0..n { for k in 0..n { c[i][j] += a[i][k] * b[k][j]; } } } c }
''',
    ],
    "O(m+n)": [
        '''fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> { let mut out = Vec::with_capacity(a.len() + b.len()); let (mut i, mut j) = (0, 0); while i < a.len() && j < b.len() { if a[i] <= b[j] { out.push(a[i]); i += 1; } else { out.push(b[j]); j += 1; } } out.extend_from_slice(&a[i..]); out.extend_from_slice(&b[j..]); out }
''',
    ],
    "O(m*n)": [
        '''fn lcs(a: &str, b: &str) -> usize { let (m, n) = (a.len(), b.len()); let ab: Vec<u8> = a.bytes().collect(); let bb: Vec<u8> = b.bytes().collect(); let mut dp = vec![vec![0usize; n+1]; m+1]; for i in 1..=m { for j in 1..=n { dp[i][j] = if ab[i-1] == bb[j-1] { dp[i-1][j-1] + 1 } else { std::cmp::max(dp[i-1][j], dp[i][j-1]) }; } } dp[m][n] }
''',
    ],
    "O(m log n)": [
        '''fn search_many(q: &[i32], s: &[i32]) -> Vec<usize> { q.iter().map(|x| s.partition_point(|&v| v < *x)).collect() }
''',
    ],
    "O((m+n) log(m+n))": [
        '''fn combined_sort(a: &[i32], b: &[i32]) -> Vec<i32> { let mut m: Vec<i32> = a.iter().chain(b.iter()).copied().collect(); m.sort(); m }
''',
    ],
}


SWIFT_TEMPLATES: dict[str, list[str]] = {
    "exponential": [
        '''func rec(_ a: [Int], _ i: Int, _ r: Int) -> Bool { if r == 0 { return true }; if i == a.count || r < 0 { return false }; return rec(a, i+1, r - a[i]) || rec(a, i+1, r) }
func subsetSum(_ a: [Int], _ t: Int) -> Bool { return rec(a, 0, t) }
''',
    ],
    "O(n^3)": [
        '''func floyd(_ dist: inout [[Int]], _ n: Int) { for k in 0..<n { for i in 0..<n { for j in 0..<n { if dist[i][k] + dist[k][j] < dist[i][j] { dist[i][j] = dist[i][k] + dist[k][j] } } } } }
''',
    ],
    "O(m+n)": [
        '''func mergeSorted(_ a: [Int], _ b: [Int]) -> [Int] { var out: [Int] = []; var i = 0; var j = 0; while i < a.count && j < b.count { if a[i] <= b[j] { out.append(a[i]); i += 1 } else { out.append(b[j]); j += 1 } }; out.append(contentsOf: a[i...]); out.append(contentsOf: b[j...]); return out }
''',
    ],
    "O(m*n)": [
        '''func lcs(_ a: String, _ b: String) -> Int { let ac = Array(a); let bc = Array(b); let m = ac.count; let n = bc.count; var dp = Array(repeating: Array(repeating: 0, count: n+1), count: m+1); for i in 1...m { for j in 1...n { dp[i][j] = ac[i-1] == bc[j-1] ? dp[i-1][j-1] + 1 : max(dp[i-1][j], dp[i][j-1]) } }; return dp[m][n] }
''',
    ],
    "O(m log n)": [
        '''func lowerBound(_ s: [Int], _ q: Int) -> Int { var lo = 0; var hi = s.count; while lo < hi { let mid = (lo + hi) / 2; if s[mid] < q { lo = mid + 1 } else { hi = mid } }; return lo }
func searchMany(_ q: [Int], _ s: [Int]) -> [Int] { return q.map { lowerBound(s, $0) } }
''',
    ],
    "O((m+n) log(m+n))": [
        '''func combinedSort(_ a: [Int], _ b: [Int]) -> [Int] { var m = a + b; m.sort(); return m }
''',
    ],
}


# Master registry
TEMPLATES_BY_LANGUAGE: dict[str, dict[str, list[str]]] = {
    "python":     PY_TEMPLATES,
    "java":       JAVA_TEMPLATES,
    "cpp":        CPP_TEMPLATES,
    "c":          C_TEMPLATES,
    "csharp":     CSHARP_TEMPLATES,
    "go":         GO_TEMPLATES,
    "javascript": JS_TEMPLATES,
    "typescript": TS_TEMPLATES,
    "php":        PHP_TEMPLATES,
    "ruby":       RUBY_TEMPLATES,
    "rust":       RUST_TEMPLATES,
    "swift":      SWIFT_TEMPLATES,
}
assert set(TEMPLATES_BY_LANGUAGE) == set(LANGUAGES), \
    f"templates drifted from LANGUAGES: missing {set(LANGUAGES) - set(TEMPLATES_BY_LANGUAGE)}"


# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------

# Word-boundary identifier renames. Per-language reserved-word filtering happens
# downstream in pipeline/07.
_RENAME_SCHEMES: tuple[dict[str, str], ...] = (
    {},
    {"i": "p", "j": "q"},
    {"x": "u", "y": "v"},
    {"out": "result"},
    {"arr": "nums", "a": "xs", "b": "ys"},
    {"dp": "table"},
    {"dist": "distances"},
    {"path": "trail"},
    {"perm": "permutation"},
    {"merged": "joined"},
    {"rec": "helper"},
)


def _rename_identifiers(src: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return src
    result = src
    for old, new in mapping.items():
        result = re.sub(rf"\b{re.escape(old)}\b", new, result)
    return result


def expand_template(language: str, label: str, idx: int, src: str,
                    max_variants: int) -> list[dict]:
    out: list[dict] = []
    for variant_i, scheme in enumerate(_RENAME_SCHEMES[:max_variants]):
        renamed = _rename_identifiers(src, scheme)
        if not syntax_ok(language, renamed):
            continue
        pid = "syn-" + hashlib.sha1(
            (language + "|" + renamed).encode("utf-8")
        ).hexdigest()[:12]
        out.append({
            "source": "synthetic",
            "language": language,
            "problem_id": pid,
            "solution_idx": variant_i,
            "code": renamed,
            "raw_complexity": label,
            "pre_label": label,
            "template": f"{language}|{label}#{idx}",
            "variant": variant_i,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/interim/parsed/supplemental.jsonl")
    ap.add_argument("--max_variants_per_template", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_emit = 0
    per_cell: dict[tuple[str, str], int] = {}
    n_template_dropped = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for language, by_label in TEMPLATES_BY_LANGUAGE.items():
            for label, templates in by_label.items():
                for idx, src in enumerate(templates):
                    # First, sanity-check the canonical template parses;
                    # if not, skip the whole template (no variants).
                    if not syntax_ok(language, src):
                        n_template_dropped += 1
                        print(f"[04] DROP template {language}|{label}#{idx}: "
                              f"failed syntax check (fix the source)", flush=True)
                        continue
                    for rec in expand_template(language, label, idx, src,
                                               args.max_variants_per_template):
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        per_cell[(language, label)] = per_cell.get((language, label), 0) + 1
                        n_emit += 1

    print(f"[04] synthetic records: {n_emit}  "
          f"(dropped templates: {n_template_dropped})", flush=True)
    for lang in LANGUAGES:
        total = sum(per_cell.get((lang, lab), 0) for lab in (
            "exponential", "O(n^3)", "O(m+n)", "O(m*n)", "O(m log n)", "O((m+n) log(m+n))"
        ))
        print(f"[04]   {lang:<12s} total={total}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
