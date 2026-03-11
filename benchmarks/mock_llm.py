"""MockLLM — a deterministic LLM backend for benchmarking without an API key.

The mock parses the incoming messages and system prompt to decide what to return:

  - If the system prompt contains "plan"      → return a structured plan
  - If the system prompt contains "execute"   → return a tool call (python_repl)
  - If the system prompt contains "reflect"   → return a reflection block
  - If the message asks for JSON hints        → return a JSON hint array
  - If the message asks for quality eval JSON → return a quality score JSON

This lets the full agent loop run end-to-end — planner, executor, reflexion,
hint extraction, quality filter — with zero API calls.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import numpy as np

from agenticmemo.llm.base import LLMBackend
from agenticmemo.types import LLMResponse, Message, ToolCall


# ---------------------------------------------------------------------------
# Python snippets the executor will "generate" and run via python_repl
# ---------------------------------------------------------------------------

_TASK_SNIPPETS: dict[str, str] = {
    # Math
    "fibonacci": (
        "def fibonacci(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n): a, b = b, a + b\n"
        "    return a\n"
        "print([fibonacci(i) for i in range(10)])"
    ),
    "prime": (
        "def is_prime(n):\n"
        "    if n < 2: return False\n"
        "    return all(n % i != 0 for i in range(2, int(n**0.5)+1))\n"
        "print([x for x in range(20) if is_prime(x)])"
    ),
    "factorial": (
        "import math\n"
        "print([math.factorial(n) for n in range(10)])"
    ),
    "sort": (
        "data = [64, 25, 12, 22, 11]\n"
        "data.sort()\n"
        "print(data)"
    ),
    "binary_search": (
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr)-1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: lo = mid + 1\n"
        "        else: hi = mid - 1\n"
        "    return -1\n"
        "arr = list(range(0, 100, 5))\n"
        "print(binary_search(arr, 35))"
    ),
    # String
    "reverse": (
        "def reverse_string(s): return s[::-1]\n"
        "print(reverse_string('hello world'))"
    ),
    "palindrome": (
        "def is_palindrome(s): return s == s[::-1]\n"
        "for w in ['racecar', 'hello', 'madam']: print(w, is_palindrome(w))"
    ),
    "word_count": (
        "text = 'the quick brown fox jumps over the lazy dog'\n"
        "from collections import Counter\n"
        "print(Counter(text.split()).most_common(5))"
    ),
    "anagram": (
        "def is_anagram(a, b): return sorted(a) == sorted(b)\n"
        "print(is_anagram('listen', 'silent'))\n"
        "print(is_anagram('hello', 'world'))"
    ),
    # Data structures
    "stack": (
        "class Stack:\n"
        "    def __init__(self): self._data = []\n"
        "    def push(self, v): self._data.append(v)\n"
        "    def pop(self): return self._data.pop()\n"
        "    def peek(self): return self._data[-1]\n"
        "s = Stack()\n"
        "for i in range(5): s.push(i)\n"
        "print([s.pop() for _ in range(5)])"
    ),
    "queue": (
        "from collections import deque\n"
        "q = deque()\n"
        "for i in range(5): q.append(i)\n"
        "print([q.popleft() for _ in range(5)])"
    ),
    # ── Hard: Dynamic Programming ─────────────────────────────────────────────
    "knapsack": (
        "weights, values, cap = [2,3,4,5], [3,4,5,6], 5\n"
        "n = len(weights)\n"
        "dp = [[0]*(cap+1) for _ in range(n+1)]\n"
        "for i in range(1,n+1):\n"
        "    for w in range(cap+1):\n"
        "        dp[i][w] = dp[i-1][w]\n"
        "        if weights[i-1] <= w:\n"
        "            dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])\n"
        "w, selected = cap, []\n"
        "for i in range(n,0,-1):\n"
        "    if dp[i][w] != dp[i-1][w]:\n"
        "        selected.append(i-1); w -= weights[i-1]\n"
        "print('Max value:', dp[n][cap])\n"
        "print('Items selected:', selected[::-1])"
    ),
    "lcs": (
        "a, b = 'ABCBDAB', 'BDCAB'\n"
        "m, n = len(a), len(b)\n"
        "dp = [[0]*(n+1) for _ in range(m+1)]\n"
        "for i in range(1,m+1):\n"
        "    for j in range(1,n+1):\n"
        "        if a[i-1]==b[j-1]: dp[i][j] = dp[i-1][j-1]+1\n"
        "        else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
        "lcs, i, j = [], m, n\n"
        "while i>0 and j>0:\n"
        "    if a[i-1]==b[j-1]: lcs.append(a[i-1]); i-=1; j-=1\n"
        "    elif dp[i-1][j]>dp[i][j-1]: i-=1\n"
        "    else: j-=1\n"
        "print('LCS length:', dp[m][n])\n"
        "print('LCS:', ''.join(reversed(lcs)))"
    ),
    "edit_distance": (
        "a, b = 'kitten', 'sitting'\n"
        "m, n = len(a), len(b)\n"
        "dp = [[0]*(n+1) for _ in range(m+1)]\n"
        "for i in range(m+1): dp[i][0] = i\n"
        "for j in range(n+1): dp[0][j] = j\n"
        "for i in range(1,m+1):\n"
        "    for j in range(1,n+1):\n"
        "        if a[i-1]==b[j-1]: dp[i][j] = dp[i-1][j-1]\n"
        "        else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n"
        "print('Edit distance:', dp[m][n])"
    ),
    "levenshtein": (
        "a, b = 'kitten', 'sitting'\n"
        "dp = list(range(len(b)+1))\n"
        "for i,ca in enumerate(a,1):\n"
        "    prev, dp[0] = dp[0], i\n"
        "    for j,cb in enumerate(b,1):\n"
        "        prev, dp[j] = dp[j], prev if ca==cb else 1+min(prev,dp[j],dp[j-1])\n"
        "print('Levenshtein distance:', dp[-1])"
    ),
    "coin_change": (
        "coins, amount = [1,5,10,25], 41\n"
        "dp = [float('inf')]*(amount+1); dp[0] = 0\n"
        "parent = [-1]*(amount+1)\n"
        "for c in coins:\n"
        "    for a in range(c, amount+1):\n"
        "        if dp[a-c]+1 < dp[a]: dp[a]=dp[a-c]+1; parent[a]=c\n"
        "used, a = [], amount\n"
        "while a>0: used.append(parent[a]); a-=parent[a]\n"
        "print('Min coins:', dp[amount])\n"
        "print('Coins used:', sorted(used, reverse=True))"
    ),
    "matrix_chain": (
        "dims = [10,30,5,60]\n"
        "n = len(dims)-1\n"
        "dp = [[0]*n for _ in range(n)]\n"
        "split = [[0]*n for _ in range(n)]\n"
        "for l in range(2,n+1):\n"
        "    for i in range(n-l+1):\n"
        "        j = i+l-1; dp[i][j] = float('inf')\n"
        "        for k in range(i,j):\n"
        "            cost = dp[i][k]+dp[k+1][j]+dims[i]*dims[k+1]*dims[j+1]\n"
        "            if cost < dp[i][j]: dp[i][j]=cost; split[i][j]=k\n"
        "print('Min multiplications:', dp[0][n-1])"
    ),
    # ── Hard: Graph ───────────────────────────────────────────────────────────
    "dijkstra": (
        "import heapq\n"
        "graph = {0:[(1,4),(2,1)], 1:[(3,1)], 2:[(1,2),(3,5)], 3:[]}\n"
        "def dijkstra(g, src):\n"
        "    dist = {n: float('inf') for n in g}; dist[src]=0\n"
        "    heap = [(0,src)]\n"
        "    while heap:\n"
        "        d,u = heapq.heappop(heap)\n"
        "        if d > dist[u]: continue\n"
        "        for v,w in g[u]:\n"
        "            if dist[u]+w < dist[v]:\n"
        "                dist[v]=dist[u]+w; heapq.heappush(heap,(dist[v],v))\n"
        "    return dist\n"
        "print(dijkstra(graph, 0))"
    ),
    "topological": (
        "graph = {0:[1,2], 1:[3], 2:[3], 3:[4], 4:[]}\n"
        "visited, stack, in_cycle = set(), [], set()\n"
        "def dfs(u):\n"
        "    in_cycle.add(u); visited.add(u)\n"
        "    for v in graph[u]:\n"
        "        if v in in_cycle: raise ValueError('Cycle!')\n"
        "        if v not in visited: dfs(v)\n"
        "    in_cycle.discard(u); stack.append(u)\n"
        "for n in graph:\n"
        "    if n not in visited: dfs(n)\n"
        "print('Topological order:', stack[::-1])"
    ),
    "word_ladder": (
        "from collections import deque\n"
        "begin, end = 'hit', 'cog'\n"
        "words = {'hot','dot','dog','lot','log','cog'}\n"
        "q = deque([(begin, [begin])])\n"
        "seen = {begin}\n"
        "while q:\n"
        "    word, path = q.popleft()\n"
        "    for i in range(len(word)):\n"
        "        for c in 'abcdefghijklmnopqrstuvwxyz':\n"
        "            nw = word[:i]+c+word[i+1:]\n"
        "            if nw in words and nw not in seen:\n"
        "                seen.add(nw); q.append((nw, path+[nw]))\n"
        "                if nw==end: print('Path:', path+[nw]); exit(0)\n"
        "print('No path found')"
    ),
    "kosaraju": (
        "graph = {0:[1], 1:[2], 2:[0,3], 3:[4], 4:[5], 5:[3]}\n"
        "rev = {u:[] for u in graph}\n"
        "for u in graph:\n"
        "    for v in graph[u]: rev[v].append(u)\n"
        "visited, order = set(), []\n"
        "def dfs1(u):\n"
        "    visited.add(u)\n"
        "    for v in graph[u]:\n"
        "        if v not in visited: dfs1(v)\n"
        "    order.append(u)\n"
        "for u in graph:\n"
        "    if u not in visited: dfs1(u)\n"
        "visited, sccs = set(), []\n"
        "def dfs2(u, scc):\n"
        "    visited.add(u); scc.append(u)\n"
        "    for v in rev[u]:\n"
        "        if v not in visited: dfs2(v, scc)\n"
        "for u in reversed(order):\n"
        "    if u not in visited:\n"
        "        scc=[]; dfs2(u,scc); sccs.append(scc)\n"
        "print('SCCs:', sccs)"
    ),
    # ── Hard: System Design ───────────────────────────────────────────────────
    "lru": (
        "class Node:\n"
        "    def __init__(self,k,v): self.k=k;self.v=v;self.prev=self.next=None\n"
        "class LRUCache:\n"
        "    def __init__(self,cap):\n"
        "        self.cap=cap; self.cache={}\n"
        "        self.head,self.tail=Node(0,0),Node(0,0)\n"
        "        self.head.next=self.tail; self.tail.prev=self.head\n"
        "    def _remove(self,n): n.prev.next=n.next; n.next.prev=n.prev\n"
        "    def _insert(self,n): n.next=self.tail; n.prev=self.tail.prev; self.tail.prev.next=n; self.tail.prev=n\n"
        "    def get(self,k):\n"
        "        if k not in self.cache: return -1\n"
        "        n=self.cache[k]; self._remove(n); self._insert(n); return n.v\n"
        "    def put(self,k,v):\n"
        "        if k in self.cache: self._remove(self.cache[k])\n"
        "        n=Node(k,v); self.cache[k]=n; self._insert(n)\n"
        "        if len(self.cache)>self.cap:\n"
        "            lru=self.head.next; self._remove(lru); del self.cache[lru.k]\n"
        "c=LRUCache(3)\n"
        "c.put(1,1);c.put(2,2);print('get(1):',c.get(1))\n"
        "c.put(3,3);c.put(4,4)\n"
        "print('get(2):',c.get(2),'get(3):',c.get(3),'get(4):',c.get(4))"
    ),
    "min_heap": (
        "class MinHeap:\n"
        "    def __init__(self): self.h=[]\n"
        "    def push(self,v):\n"
        "        self.h.append(v); i=len(self.h)-1\n"
        "        while i>0:\n"
        "            p=(i-1)//2\n"
        "            if self.h[p]>self.h[i]: self.h[p],self.h[i]=self.h[i],self.h[p]; i=p\n"
        "            else: break\n"
        "    def pop(self):\n"
        "        if len(self.h)==1: return self.h.pop()\n"
        "        top=self.h[0]; self.h[0]=self.h.pop(); i=0\n"
        "        while True:\n"
        "            l,r,s=2*i+1,2*i+2,i\n"
        "            if l<len(self.h) and self.h[l]<self.h[s]: s=l\n"
        "            if r<len(self.h) and self.h[r]<self.h[s]: s=r\n"
        "            if s==i: break\n"
        "            self.h[i],self.h[s]=self.h[s],self.h[i]; i=s\n"
        "        return top\n"
        "h=MinHeap()\n"
        "for x in [5,3,8,1,9,2,7,4,6]: h.push(x)\n"
        "print([h.pop() for _ in range(9)])"
    ),
    "trie": (
        "class TrieNode:\n"
        "    def __init__(self): self.children={}; self.end=False\n"
        "class Trie:\n"
        "    def __init__(self): self.root=TrieNode()\n"
        "    def insert(self,w):\n"
        "        n=self.root\n"
        "        for c in w: n=n.children.setdefault(c,TrieNode())\n"
        "        n.end=True\n"
        "    def search(self,w):\n"
        "        n=self.root\n"
        "        for c in w:\n"
        "            if c not in n.children: return False\n"
        "            n=n.children[c]\n"
        "        return n.end\n"
        "    def starts_with(self,p):\n"
        "        n=self.root\n"
        "        for c in p:\n"
        "            if c not in n.children: return []\n"
        "            n=n.children[c]\n"
        "        res=[]; self._collect(n,list(p),res); return res\n"
        "    def _collect(self,n,path,res):\n"
        "        if n.end: res.append(''.join(path))\n"
        "        for c,ch in n.children.items(): self._collect(ch,path+[c],res)\n"
        "t=Trie()\n"
        "for w in ['apple','app','application','apply','apt']: t.insert(w)\n"
        "print('search app:',t.search('app'))\n"
        "print('search ap:',t.search('ap'))\n"
        "print('starts_with app:',t.starts_with('app'))"
    ),
    "sliding_window": (
        "from collections import deque\n"
        "def max_sliding_window(arr, k):\n"
        "    dq, res = deque(), []\n"
        "    for i,v in enumerate(arr):\n"
        "        while dq and dq[0] < i-k+1: dq.popleft()\n"
        "        while dq and arr[dq[-1]] < v: dq.pop()\n"
        "        dq.append(i)\n"
        "        if i >= k-1: res.append(arr[dq[0]])\n"
        "    return res\n"
        "print(max_sliding_window([2,1,5,3,6,4,8,2], 3))"
    ),
    "rate_limiter": (
        "from collections import deque\n"
        "class SlidingWindowRateLimiter:\n"
        "    def __init__(self,limit,window): self.limit=limit; self.window=window; self.q=deque()\n"
        "    def allow(self,t):\n"
        "        while self.q and self.q[0] <= t-self.window: self.q.popleft()\n"
        "        if len(self.q) < self.limit: self.q.append(t); return True\n"
        "        return False\n"
        "rl = SlidingWindowRateLimiter(3, 10)\n"
        "for t in [0,1,2,5,9,10,11,15]:\n"
        "    print(f't={t}: {\"ALLOW\" if rl.allow(t) else \"DENY\"}')"
    ),
    # ── Hard: Algorithms ──────────────────────────────────────────────────────
    "merge_sort": (
        "def merge_sort_count(arr):\n"
        "    if len(arr)<=1: return arr,0\n"
        "    mid=len(arr)//2\n"
        "    l,lc=merge_sort_count(arr[:mid])\n"
        "    r,rc=merge_sort_count(arr[mid:])\n"
        "    merged,inv,i,j=[],lc+rc,0,0\n"
        "    while i<len(l) and j<len(r):\n"
        "        if l[i]<=r[j]: merged.append(l[i]); i+=1\n"
        "        else: merged.append(r[j]); j+=1; inv+=len(l)-i\n"
        "    return merged+l[i:]+r[j:], inv\n"
        "arr=[8,4,2,1,6,3,5,7]\n"
        "sorted_arr,inv=merge_sort_count(arr)\n"
        "print('Sorted:', sorted_arr)\n"
        "print('Inversions:', inv)"
    ),
    "n_queens": (
        "def solve_nqueens(n):\n"
        "    solutions=[]; cols=set(); d1=set(); d2=set()\n"
        "    board=[['.']*n for _ in range(n)]\n"
        "    def bt(r):\n"
        "        if r==n: solutions.append([''.join(row) for row in board]); return\n"
        "        for c in range(n):\n"
        "            if c in cols or r-c in d1 or r+c in d2: continue\n"
        "            cols.add(c); d1.add(r-c); d2.add(r+c); board[r][c]='Q'\n"
        "            bt(r+1)\n"
        "            cols.discard(c); d1.discard(r-c); d2.discard(r+c); board[r][c]='.'\n"
        "    bt(0); return solutions\n"
        "sols=solve_nqueens(6)\n"
        "print('Solutions:', len(sols))\n"
        "print('One solution:'); [print(row) for row in sols[0]]"
    ),
    "huffman": (
        "import heapq\n"
        "from collections import Counter\n"
        "text='abracadabra'\n"
        "freq=Counter(text)\n"
        "heap=[[f,[c,'']] for c,f in freq.items()]\n"
        "heapq.heapify(heap)\n"
        "while len(heap)>1:\n"
        "    l=heapq.heappop(heap); r=heapq.heappop(heap)\n"
        "    for p in l[1:]: p[1]='0'+p[1]\n"
        "    for p in r[1:]: p[1]='1'+p[1]\n"
        "    heapq.heappush(heap,[l[0]+r[0]]+l[1:]+r[1:])\n"
        "codes={p[0]:p[1] for p in heap[0][1:]}\n"
        "encoded=''.join(codes[c] for c in text)\n"
        "ratio=len(encoded)/(len(text)*8)\n"
        "print('Codes:', codes)\n"
        "print('Encoded len:', len(encoded))\n"
        "print('Compression ratio:', round(ratio,3))"
    ),
    "expression_eval": (
        "def evaluate(s):\n"
        "    nums, ops = [], []\n"
        "    prec = {'+':1,'-':1,'*':2,'/':2}\n"
        "    def apply():\n"
        "        b,a = nums.pop(),nums.pop()\n"
        "        op = ops.pop()\n"
        "        if op=='+': nums.append(a+b)\n"
        "        elif op=='-': nums.append(a-b)\n"
        "        elif op=='*': nums.append(a*b)\n"
        "        else: nums.append(int(a/b))\n"
        "    i=0\n"
        "    while i<len(s):\n"
        "        c=s[i]\n"
        "        if c.isdigit():\n"
        "            n=0\n"
        "            while i<len(s) and s[i].isdigit(): n=n*10+int(s[i]); i+=1\n"
        "            nums.append(n); continue\n"
        "        elif c=='(': ops.append(c)\n"
        "        elif c==')':\n"
        "            while ops[-1]!='(': apply()\n"
        "            ops.pop()\n"
        "        elif c in prec:\n"
        "            while ops and ops[-1] in prec and prec[ops[-1]]>=prec[c]: apply()\n"
        "            ops.append(c)\n"
        "        i+=1\n"
        "    while ops: apply()\n"
        "    return nums[0]\n"
        "print(evaluate('(3+5)*2-4/(2+2)'))\n"
        "print(evaluate('((2+3)*4-(6/2))'))"
    ),
    # ── Hard repeat tasks ─────────────────────────────────────────────────────
    "lfu": (
        "from collections import defaultdict\n"
        "class LFUCache:\n"
        "    def __init__(self,cap):\n"
        "        self.cap=cap; self.min_freq=0\n"
        "        self.key_val={}; self.key_freq={}; self.freq_keys=defaultdict(dict)\n"
        "    def _update(self,k):\n"
        "        f=self.key_freq[k]; self.key_freq[k]=f+1\n"
        "        del self.freq_keys[f][k]\n"
        "        if not self.freq_keys[f] and f==self.min_freq: self.min_freq+=1\n"
        "        self.freq_keys[f+1][k]=None\n"
        "    def get(self,k):\n"
        "        if k not in self.key_val: return -1\n"
        "        self._update(k); return self.key_val[k]\n"
        "    def put(self,k,v):\n"
        "        if not self.cap: return\n"
        "        if k in self.key_val: self._update(k); self.key_val[k]=v; return\n"
        "        if len(self.key_val)>=self.cap:\n"
        "            evict=next(iter(self.freq_keys[self.min_freq]))\n"
        "            del self.freq_keys[self.min_freq][evict],self.key_val[evict],self.key_freq[evict]\n"
        "        self.key_val[k]=v; self.key_freq[k]=1; self.freq_keys[1][k]=None; self.min_freq=1\n"
        "c=LFUCache(3)\n"
        "c.put(1,1);c.put(2,2);c.put(3,3);print('get(1):',c.get(1))\n"
        "c.put(4,4);print('get(2):',c.get(2),'get(3):',c.get(3))"
    ),
    "bellman_ford": (
        "def bellman_ford(n, edges, src):\n"
        "    dist=[float('inf')]*n; dist[src]=0\n"
        "    for _ in range(n-1):\n"
        "        for u,v,w in edges:\n"
        "            if dist[u]+w < dist[v]: dist[v]=dist[u]+w\n"
        "    for u,v,w in edges:\n"
        "        if dist[u]+w < dist[v]: return None  # negative cycle\n"
        "    return dist\n"
        "edges=[(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)]\n"
        "print(bellman_ford(4, edges, 0))"
    ),
    "lis": (
        "import bisect\n"
        "def lis(arr):\n"
        "    tails, parent, idx_map = [], [-1]*len(arr), {}\n"
        "    for i,x in enumerate(arr):\n"
        "        pos=bisect.bisect_left(tails,x)\n"
        "        if pos==len(tails): tails.append(x)\n"
        "        else: tails[pos]=x\n"
        "        idx_map[pos]=i\n"
        "        parent[i]=idx_map.get(pos-1,-1)\n"
        "    seq,i=[],idx_map[len(tails)-1]\n"
        "    while i!=-1: seq.append(arr[i]); i=parent[i]\n"
        "    return len(tails), seq[::-1]\n"
        "l, seq = lis([10,9,2,5,3,7,101,18])\n"
        "print('LIS length:', l)\n"
        "print('LIS:', seq)"
    ),
    "unbounded_knapsack": (
        "def unbounded_knapsack(weights, values, cap):\n"
        "    dp = [0]*(cap+1)\n"
        "    choice = [0]*(cap+1)\n"
        "    for w in range(1, cap+1):\n"
        "        for i,(wt,v) in enumerate(zip(weights,values)):\n"
        "            if wt<=w and dp[w-wt]+v > dp[w]:\n"
        "                dp[w]=dp[w-wt]+v; choice[w]=i\n"
        "    items, w = [], cap\n"
        "    while w>0 and choice[w]>=0:\n"
        "        items.append(choice[w]); w-=weights[choice[w]]\n"
        "    return dp[cap], items\n"
        "val, items = unbounded_knapsack([1,3,4,5],[1,4,5,7],7)\n"
        "print('Max value:', val)\n"
        "print('Items:', items)"
    ),
    "quicksort": (
        "def quicksort_3way(arr, lo, hi):\n"
        "    if lo>=hi: return\n"
        "    lt,gt,i,pivot=lo,hi,lo,arr[lo]\n"
        "    while i<=gt:\n"
        "        if arr[i]<pivot: arr[lt],arr[i]=arr[i],arr[lt]; lt+=1; i+=1\n"
        "        elif arr[i]>pivot: arr[gt],arr[i]=arr[i],arr[gt]; gt-=1\n"
        "        else: i+=1\n"
        "    quicksort_3way(arr,lo,lt-1); quicksort_3way(arr,gt+1,hi)\n"
        "arr=[3,6,8,10,1,2,1,3,6,8,3]\n"
        "quicksort_3way(arr,0,len(arr)-1)\n"
        "print('Sorted:', arr)"
    ),
    # ── Hard failure tasks ────────────────────────────────────────────────────
    "regex_match": (
        "def is_match(s, p):\n"
        "    m,n=len(s),len(p)\n"
        "    dp=[[False]*(n+1) for _ in range(m+1)]\n"
        "    dp[0][0]=True\n"
        "    for j in range(2,n+1):\n"
        "        if p[j-1]=='*': dp[0][j]=dp[0][j-2]\n"
        "    for i in range(1,m+1):\n"
        "        for j in range(1,n+1):\n"
        "            if p[j-1]=='*':\n"
        "                dp[i][j]=dp[i][j-2] or (dp[i-1][j] and (p[j-2]=='.' or p[j-2]==s[i-1]))\n"
        "            elif p[j-1]=='.' or p[j-1]==s[i-1]: dp[i][j]=dp[i-1][j-1]\n"
        "    return dp[m][n]\n"
        "tests=[('aa','a*',True),('ab','.*',True),('aab','c*a*b',True),('mississippi','mis*is*p*.',False)]\n"
        "for s,p,exp in tests: r=is_match(s,p); print(f'{s!r},{p!r}: {r} ({\"OK\" if r==exp else \"FAIL\"})')"
    ),
    "rain_water": (
        "def trap(h):\n"
        "    l,r,lm,rm,res=0,len(h)-1,0,0,0\n"
        "    while l<r:\n"
        "        if h[l]<h[r]:\n"
        "            if h[l]>=lm: lm=h[l]\n"
        "            else: res+=lm-h[l]\n"
        "            l+=1\n"
        "        else:\n"
        "            if h[r]>=rm: rm=h[r]\n"
        "            else: res+=rm-h[r]\n"
        "            r-=1\n"
        "    return res\n"
        "print(trap([0,1,0,2,1,0,1,3,2,1,2,1]))"
    ),
    "thread_safe": (
        "import threading, time, random\n"
        "class BoundedBlockingQueue:\n"
        "    def __init__(self,cap): self.cap=cap; self.q=[]; self.lock=threading.Lock(); self.not_full=threading.Condition(self.lock); self.not_empty=threading.Condition(self.lock)\n"
        "    def enqueue(self,v):\n"
        "        with self.not_full:\n"
        "            while len(self.q)>=self.cap: self.not_full.wait()\n"
        "            self.q.append(v); self.not_empty.notify()\n"
        "    def dequeue(self):\n"
        "        with self.not_empty:\n"
        "            while not self.q: self.not_empty.wait()\n"
        "            v=self.q.pop(0); self.not_full.notify(); return v\n"
        "bq=BoundedBlockingQueue(3)\n"
        "results=[]\n"
        "def producer(i):\n"
        "    bq.enqueue(i); results.append(f'P{i} enqueued')\n"
        "def consumer():\n"
        "    v=bq.dequeue(); results.append(f'consumed {v}')\n"
        "ts=[threading.Thread(target=producer,args=(i,)) for i in range(5)]+[threading.Thread(target=consumer) for _ in range(5)]\n"
        "for t in ts: t.start()\n"
        "for t in ts: t.join()\n"
        "for r in sorted(results): print(r)"
    ),
    # Default fallback
    "default": (
        "result = 'Task completed successfully'\n"
        "print(result)"
    ),
}


def _pick_snippet(task: str) -> str:
    """Pick the best matching code snippet for the task description."""
    task_lower = task.lower()
    for keyword, snippet in _TASK_SNIPPETS.items():
        if keyword in task_lower:
            return snippet
    return _TASK_SNIPPETS["default"]


class MockLLM(LLMBackend):
    """Deterministic mock LLM for benchmark and integration testing.

    Produces realistic-looking responses for every stage of the agent loop
    without making any network requests.

    Args:
        fail_rate: Probability [0,1] that a task "fails" on first attempt,
                   triggering the Reflexion loop. Useful for testing recovery.
        latency_ms: Simulated response latency in milliseconds.
    """

    def __init__(
        self,
        model: str = "mock-llm-v1",
        fail_rate: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(model=model, temperature=0.0, max_tokens=1024)
        self.fail_rate = fail_rate
        self.latency_ms = latency_ms
        self._call_count = 0
        self._token_budget = 0   # simulated tokens

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        import asyncio  # noqa: PLC0415
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        self._call_count += 1
        system_lower = (system or "").lower()
        last_content = messages[-1].content if messages else ""

        # ---- Hint extraction request ----------------------------------------
        if "internalized hints" in last_content.lower() or \
           "extract" in last_content.lower() and "hint" in last_content.lower():
            return self._hint_response()

        # ---- Quality self-evaluation ----------------------------------------
        if '"coherent"' in last_content or "quality evaluator" in last_content.lower():
            return self._quality_response()

        # ---- Reflexion (failure diagnosis) ----------------------------------
        if "diagnosis:" in system_lower or "reflection" in system_lower or \
           "failed trajectory" in last_content.lower():
            return self._reflexion_response()

        # ---- Planner (step-by-step plan) ------------------------------------
        if "plan" in system_lower or "step-by-step" in last_content.lower():
            task = self._extract_task(last_content)
            return self._plan_response(task, tools)

        # ---- Executor (tool call or final answer) ---------------------------
        if tools:
            task = self._extract_task(last_content)
            # Simulate occasional failure for reflexion testing
            import random  # noqa: PLC0415
            if random.random() < self.fail_rate:
                return self._final_answer_response(success=False)
            return self._tool_call_response(task)

        return self._final_answer_response(success=True)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic pseudo-embeddings based on text hash."""
        results = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(384).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            results.append(vec.tolist())
        return results

    # ------------------------------------------------------------------ #
    # Response builders
    # ------------------------------------------------------------------ #

    def _plan_response(self, task: str, tools: list | None) -> LLMResponse:
        tool_name = "python_repl"
        snippet = _pick_snippet(task).split("\n")[0]
        content = (
            f"Step 1: Write the solution code → {tool_name}(code=\"{snippet}...\")\n"
            f"Step 2: Verify the output is correct → {tool_name}(code=\"print('done')\")\n"
            f"Step 3: Return the final answer"
        )
        return LLMResponse(
            content=content, model=self.model,
            input_tokens=len(content) // 4, output_tokens=len(content) // 4,
        )

    def _tool_call_response(self, task: str) -> LLMResponse:
        code = _pick_snippet(task)
        return LLMResponse(
            content=f"Executing solution for: {task[:60]}",
            tool_calls=[ToolCall(
                id=str(uuid.uuid4()),
                name="python_repl",
                arguments={"code": code},
            )],
            model=self.model,
            input_tokens=50,
            output_tokens=len(code) // 4,
        )

    def _final_answer_response(self, success: bool = True) -> LLMResponse:
        if success:
            content = "Task completed successfully. The implementation is correct and handles edge cases."
        else:
            content = "I was unable to complete the task. The approach did not produce the expected result."
        return LLMResponse(
            content=content, model=self.model,
            input_tokens=30, output_tokens=20, stop_reason="end_turn",
        )

    def _reflexion_response(self) -> LLMResponse:
        content = (
            "DIAGNOSIS: The initial approach did not correctly handle edge cases.\n"
            "MISTAKES: Missing boundary check for empty input; incorrect loop termination.\n"
            "CORRECTION: Add input validation at the start; use inclusive upper bound in range."
        )
        return LLMResponse(content=content, model=self.model, input_tokens=40, output_tokens=40)

    def _hint_response(self) -> LLMResponse:
        hints = [
            "Always validate inputs before processing (check for None, empty, out-of-range).",
            "Test with boundary values: empty collections, zero, negative numbers.",
            "Use built-in Python libraries (itertools, collections, math) for efficiency.",
            "Print intermediate results during debugging to trace logic errors.",
            "Break complex tasks into helper functions for clarity and reusability.",
        ]
        return LLMResponse(
            content=json.dumps(hints), model=self.model,
            input_tokens=80, output_tokens=60,
        )

    def _quality_response(self) -> LLMResponse:
        score = {"coherent": True, "correct": True, "efficient": True, "score": 0.85}
        return LLMResponse(
            content=json.dumps(score), model=self.model,
            input_tokens=60, output_tokens=20,
        )

    @staticmethod
    def _extract_task(content: str) -> str:
        """Pull the task string from a message."""
        for prefix in ("Execute this task: ", "TASK: ", "Task: "):
            if prefix in content:
                idx = content.index(prefix) + len(prefix)
                return content[idx:idx+120].split("\n")[0]
        return content[:80]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    @property
    def call_count(self) -> int:
        return self._call_count
