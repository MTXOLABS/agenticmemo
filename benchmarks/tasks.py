"""Benchmark task definitions for all four phases.

Tasks are grouped by category. Each task is a plain English description
that the agent receives. The MockLLM maps these to real executable Python
via keyword matching.

Task groups:
  MATH_TASKS       — algorithms, numerics
  STRING_TASKS     — text manipulation
  DS_TASKS         — data structures
  REPEAT_TASKS     — used to test memory accumulation (similar to earlier tasks)
  FAILURE_TASKS    — tasks the MockLLM may fail on (for Reflexion testing)
  MULTIAGENT_TASKS — tasks split between researcher + coder agents

Hard task groups (for --hard research mode):
  HARD_DP_TASKS      — dynamic programming problems
  HARD_GRAPH_TASKS   — graph algorithms
  HARD_SYSDESIGN_TASKS — system design / advanced data structures
  HARD_ALGO_TASKS    — complex algorithmic problems
  HARD_REPEAT_TASKS  — transfer: similar to hard tasks but lexically different
  HARD_FAILURE_TASKS — deliberately tricky, trigger Reflexion
"""

from dataclasses import dataclass, field


@dataclass
class BenchTask:
    task: str
    category: str
    expected_keywords: list[str] = field(default_factory=list)
    domain: str = "coding"


MATH_TASKS: list[BenchTask] = [
    BenchTask(
        "Write a Python function that computes the nth Fibonacci number iteratively and print the first 10 values.",
        "math", ["fibonacci", "function"],
    ),
    BenchTask(
        "Write a Python function to check if a number is prime, then test it with numbers 1 to 20.",
        "math", ["prime", "function"],
    ),
    BenchTask(
        "Compute the factorial of numbers 0 through 9 using Python's math module.",
        "math", ["factorial"],
    ),
]

STRING_TASKS: list[BenchTask] = [
    BenchTask(
        "Write a Python function to reverse a string and test it with 'hello world'.",
        "string", ["reverse"],
    ),
    BenchTask(
        "Write a Python function to check if a word is a palindrome and test with 'racecar', 'hello', 'madam'.",
        "string", ["palindrome"],
    ),
    BenchTask(
        "Count the 5 most frequent words in the sentence 'the quick brown fox jumps over the lazy dog'.",
        "string", ["word_count", "counter"],
    ),
    BenchTask(
        "Write a Python function to check if two strings are anagrams. Test with 'listen'/'silent' and 'hello'/'world'.",
        "string", ["anagram"],
    ),
]

DS_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement a Stack class with push, pop, and peek methods. Push 0-4, then pop all and print.",
        "data_structures", ["stack", "push", "pop"],
    ),
    BenchTask(
        "Implement a FIFO queue using collections.deque. Enqueue 0-4, then dequeue all and print.",
        "data_structures", ["queue", "deque"],
    ),
    BenchTask(
        "Sort the list [64, 25, 12, 22, 11] in ascending order and print the result.",
        "data_structures", ["sort"],
    ),
    BenchTask(
        "Implement binary search on a sorted list. Find the index of 35 in range(0, 100, 5).",
        "data_structures", ["binary_search"],
    ),
]

# Similar to math/string tasks — used to test that memory from earlier tasks helps here
REPEAT_TASKS: list[BenchTask] = [
    BenchTask(
        "Write another Fibonacci function, this time using recursion, and compute fib(10).",
        "math", ["fibonacci"],
    ),
    BenchTask(
        "Implement the Sieve of Eratosthenes to find all prime numbers below 50.",
        "math", ["prime"],
    ),
    BenchTask(
        "Write a Python function to reverse words in a sentence (not individual characters).",
        "string", ["reverse"],
    ),
]

# These trigger Reflexion on fail_rate > 0
FAILURE_TASKS: list[BenchTask] = [
    BenchTask(
        "Write a Python generator that yields Fibonacci numbers indefinitely.",
        "math", ["fibonacci"],
    ),
    BenchTask(
        "Write a recursive palindrome checker that handles None and empty string inputs.",
        "string", ["palindrome"],
    ),
]

# For Phase 4 multi-agent benchmarking
MULTIAGENT_TASKS: list[tuple[str, BenchTask]] = [
    ("researcher", BenchTask("Explain what binary search is and when to use it.", "research")),
    ("coder",      BenchTask("Implement binary search in Python and test it.", "coding", ["binary_search"])),
    ("researcher", BenchTask("Describe the time complexity of merge sort.", "research")),
    ("coder",      BenchTask("Sort the list [64, 25, 12, 22, 11] and print the result.", "coding", ["sort"])),
    ("researcher", BenchTask("What is a palindrome?", "research")),
    ("coder",      BenchTask("Write a function to check if a word is a palindrome.", "coding", ["palindrome"])),
]

ALL_SOLO_TASKS: list[BenchTask] = MATH_TASKS + STRING_TASKS + DS_TASKS
ALL_REPEAT_TASKS: list[BenchTask] = REPEAT_TASKS


# ═══════════════════════════════════════════════════════════════════════════
# HARD BENCHMARK TASKS — genuinely challenging algorithm/system problems
# ═══════════════════════════════════════════════════════════════════════════

HARD_DP_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement 0/1 Knapsack using dynamic programming in Python. "
        "Given weights=[2,3,4,5], values=[3,4,5,6], capacity=5, "
        "find the maximum value and print which items were selected.",
        "dp", ["knapsack", "dp"],
    ),
    BenchTask(
        "Find the Longest Common Subsequence (LCS) of 'ABCBDAB' and 'BDCAB' using DP. "
        "Print the length and reconstruct the actual subsequence.",
        "dp", ["lcs", "longest_common"],
    ),
    BenchTask(
        "Implement the edit distance (Levenshtein distance) algorithm between "
        "'kitten' and 'sitting'. Print the DP table and final distance.",
        "dp", ["edit_distance", "levenshtein"],
    ),
    BenchTask(
        "Solve coin change problem: given coins=[1,5,10,25] and amount=41, "
        "find the minimum number of coins and list which coins are used.",
        "dp", ["coin_change", "coins"],
    ),
    BenchTask(
        "Implement matrix chain multiplication DP to find the optimal parenthesization "
        "for matrices with dimensions [10,30,5,60]. Print the minimum multiplications.",
        "dp", ["matrix_chain"],
    ),
]

HARD_GRAPH_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement Dijkstra's shortest path algorithm from scratch using a priority queue. "
        "Find shortest paths from node 0 in graph: "
        "{0:[(1,4),(2,1)], 1:[(3,1)], 2:[(1,2),(3,5)], 3:[]}. "
        "Print distances to all nodes.",
        "graph", ["dijkstra", "shortest_path"],
    ),
    BenchTask(
        "Implement topological sort using DFS on a DAG. "
        "Graph: {0:[1,2], 1:[3], 2:[3], 3:[4], 4:[]}. "
        "Detect if a cycle exists; if not, print topological order.",
        "graph", ["topological_sort", "topo"],
    ),
    BenchTask(
        "Implement BFS word ladder: find shortest transformation from 'hit' to 'cog' "
        "using word list ['hot','dot','dog','lot','log','cog']. "
        "Each step must change exactly one letter. Print path and steps.",
        "graph", ["word_ladder", "bfs"],
    ),
    BenchTask(
        "Find all strongly connected components in the directed graph using Kosaraju's algorithm: "
        "{0:[1], 1:[2], 2:[0,3], 3:[4], 4:[5], 5:[3]}. Print each SCC.",
        "graph", ["scc", "kosaraju", "strongly_connected"],
    ),
]

HARD_SYSDESIGN_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement an LRU (Least Recently Used) cache from scratch using a doubly-linked list "
        "and hashmap (NOT OrderedDict). Capacity=3. "
        "Perform: put(1,1), put(2,2), get(1), put(3,3), put(4,4), get(2), get(3), get(4). "
        "Print result of each get.",
        "system_design", ["lru", "lru_cache"],
    ),
    BenchTask(
        "Implement a Min-Heap from scratch (no heapq). Include: insert, extract_min, "
        "heapify. Insert [5,3,8,1,9,2,7,4,6]. Extract all elements in sorted order.",
        "system_design", ["min_heap", "heap"],
    ),
    BenchTask(
        "Implement a Trie (prefix tree) with insert, search, and starts_with methods. "
        "Insert words: ['apple','app','application','apply','apt']. "
        "Test: search('app') -> True, search('ap') -> False, starts_with('app') -> all matches.",
        "system_design", ["trie", "prefix_tree"],
    ),
    BenchTask(
        "Implement a sliding window maximum: given array [2,1,5,3,6,4,8,2] and window k=3, "
        "find max in each window using a deque (O(n) solution). Print all window maxima.",
        "system_design", ["sliding_window", "deque_max"],
    ),
    BenchTask(
        "Implement a rate limiter using the sliding window counter algorithm. "
        "Allow max 3 requests per 10-second window. Simulate 8 requests at times "
        "[0,1,2,5,9,10,11,15] seconds. Print ALLOW/DENY for each.",
        "system_design", ["rate_limiter", "sliding_window"],
    ),
]

HARD_ALGO_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement merge sort and count the number of inversions in [8,4,2,1,6,3,5,7]. "
        "An inversion is a pair (i,j) where i<j but arr[i]>arr[j]. "
        "Print sorted array and inversion count.",
        "algo", ["merge_sort", "inversions"],
    ),
    BenchTask(
        "Solve the N-Queens problem for N=6 using backtracking. "
        "Print the total number of solutions and one valid board configuration.",
        "algo", ["n_queens", "backtracking"],
    ),
    BenchTask(
        "Implement Huffman encoding: given text 'abracadabra', "
        "build the Huffman tree, assign codes to each character, "
        "encode the text and print codes + compression ratio vs 8-bit ASCII.",
        "algo", ["huffman", "huffman_encoding"],
    ),
    BenchTask(
        "Implement a stack-based expression evaluator that handles +,-,*,/ and parentheses. "
        "Evaluate: '(3+5)*2-4/(2+2)' and '((2+3)*4-(6/2))'. Print results.",
        "algo", ["expression_eval", "stack_eval"],
    ),
]

# Hard repeat/transfer tasks — structurally similar to hard tasks above
# AgenticMemo should retrieve LRU → help LFU, Dijkstra → help Bellman-Ford, etc.
HARD_REPEAT_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement an LFU (Least Frequently Used) cache with capacity=3. "
        "put(1,1), put(2,2), put(3,3), get(1), put(4,4), get(2), get(3). "
        "Print result of each get. (Hint: use frequency buckets.)",
        "system_design", ["lfu", "lfu_cache"],
    ),
    BenchTask(
        "Implement Bellman-Ford shortest path algorithm. "
        "Graph edges: [(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)]. "
        "Find shortest paths from node 0. Detect negative cycles.",
        "graph", ["bellman_ford"],
    ),
    BenchTask(
        "Implement the longest increasing subsequence (LIS) of [10,9,2,5,3,7,101,18] "
        "using DP with O(n log n) patience sorting. Print length and the subsequence.",
        "dp", ["lis", "longest_increasing"],
    ),
    BenchTask(
        "Solve the 0/1 knapsack variant: unbounded knapsack where each item can be used "
        "multiple times. weights=[1,3,4,5], values=[1,4,5,7], capacity=7. "
        "Print max value and item selection.",
        "dp", ["unbounded_knapsack"],
    ),
    BenchTask(
        "Implement quicksort with 3-way partitioning (Dutch National Flag) on "
        "[3,6,8,10,1,2,1,3,6,8,3]. Print sorted result and compare with regular quicksort "
        "step count on arrays with many duplicates.",
        "algo", ["quicksort", "3way_partition"],
    ),
]

# Hard failure tasks — genuinely tricky edge cases that cause LLMs to fail
HARD_FAILURE_TASKS: list[BenchTask] = [
    BenchTask(
        "Implement regex matching with '.' (any char) and '*' (zero or more of preceding) "
        "using DP. Test: match('aa','a*')==True, match('ab','.*')==True, "
        "match('aab','c*a*b')==True, match('mississippi','mis*is*p*.')==False.",
        "algo", ["regex_match", "dp"],
    ),
    BenchTask(
        "Implement the trapping rain water problem: given heights=[0,1,0,2,1,0,1,3,2,1,2,1], "
        "compute total trapped water using O(1) space two-pointer approach. "
        "Print result (answer: 6).",
        "algo", ["rain_water", "two_pointer"],
    ),
    BenchTask(
        "Implement a thread-safe bounded blocking queue using threading.Lock and "
        "threading.Condition. Capacity=3. Simulate 5 producers and 3 consumers "
        "running concurrently. Print enqueue/dequeue events.",
        "system_design", ["thread_safe", "blocking_queue"],
    ),
]

ALL_HARD_TASKS: list[BenchTask] = (
    HARD_DP_TASKS + HARD_GRAPH_TASKS + HARD_SYSDESIGN_TASKS + HARD_ALGO_TASKS
)
ALL_HARD_REPEAT_TASKS: list[BenchTask] = HARD_REPEAT_TASKS
