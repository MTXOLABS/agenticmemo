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


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-DOMAIN BENCHMARK TASKS
# ═══════════════════════════════════════════════════════════════════════════

FINANCE_TASKS: list[BenchTask] = [
    BenchTask(
        "Build a full WACC model then DCF valuation: "
        "Equity=$600M (beta=1.3, risk-free=4.5%, market premium=6%), Debt=$400M (pre-tax cost=7%, tax=28%). "
        "Project FCF growing at 18%/yr for 5 years from base FCF=$85M, then terminal growth=3.5%. "
        "Compute WACC, each year's FCF, terminal value, enterprise value, and equity value. "
        "Also compute EV/EBITDA assuming EBITDA=$120M. Print all intermediate steps.",
        "finance", ["wacc", "dcf", "terminal_value", "ev_ebitda"], domain="finance",
    ),
    BenchTask(
        "Implement full Markowitz mean-variance portfolio optimization WITHOUT scipy. "
        "4 assets: returns=[0.12,0.18,0.09,0.15], "
        "covariance matrix=[[0.04,0.012,0.008,0.015],[0.012,0.09,0.01,0.02],[0.008,0.01,0.025,0.007],[0.015,0.02,0.007,0.06]]. "
        "Use gradient descent to find: (1) minimum variance portfolio, (2) maximum Sharpe portfolio (rf=4%), "
        "(3) portfolio targeting 13% return. Print weights, expected return, volatility, Sharpe for each. "
        "Enforce weights sum to 1 and are non-negative.",
        "finance", ["markowitz", "efficient_frontier", "sharpe", "gradient_descent"], domain="finance",
    ),
    BenchTask(
        "Implement a full Monte Carlo VaR and CVaR engine: "
        "Portfolio of 3 correlated assets, weights=[0.5,0.3,0.2], daily returns mu=[0.0005,0.0008,0.0003], "
        "correlation matrix=[[1,0.6,0.3],[0.6,1,0.4],[0.3,0.4,1]], daily vols=[0.015,0.022,0.010]. "
        "Simulate 50,000 paths over 10-day horizon using Cholesky decomposition. "
        "Compute 95% and 99% VaR and CVaR (Expected Shortfall). "
        "Also compute Component VaR for each asset. Portfolio value=$5M. Print all results.",
        "finance", ["monte_carlo", "var", "cvar", "cholesky", "component_var"], domain="finance",
    ),
    BenchTask(
        "Build a complete fixed income analytics system: "
        "Bond: face=$1000, coupon=6% semi-annual, maturity=10 years, YTM=7.5%. "
        "Compute: (1) dirty price and clean price, (2) Macaulay duration, (3) modified duration, "
        "(4) convexity, (5) price change for +100bps and -100bps shock using duration+convexity approximation vs exact reprice, "
        "(6) DV01. Then build an immunization portfolio using 2-year (YTM=5%) and 15-year (YTM=8%) bonds "
        "to immunize a $1M liability due in 10 years. Print weights and verify duration match.",
        "finance", ["duration", "convexity", "immunization", "dv01", "bond_pricing"], domain="finance",
    ),
    BenchTask(
        "Implement a pairs trading strategy with statistical arbitrage: "
        "Generate 252 days of correlated price series for stocks A and B (seed=42): "
        "A follows GBM(mu=0.08,sigma=0.2), B=0.85*A_return + noise(sigma=0.05), both start at $50. "
        "Step 1: Test cointegration using Engle-Granger (compute ADF statistic on spread). "
        "Step 2: Compute hedge ratio via OLS regression. "
        "Step 3: Trade when spread > 2std (short spread) or < -2std (long spread), exit at mean. "
        "Step 4: Calculate Sharpe ratio, max drawdown, win rate, total P&L. Print full stats.",
        "finance", ["pairs_trading", "cointegration", "adf", "hedge_ratio", "sharpe"], domain="finance",
    ),
    BenchTask(
        "Build a multi-factor credit risk model (Merton structural model): "
        "Firm assets V0=$150M, asset volatility sigma_V=0.25, debt face value=$100M due in T=3 years, "
        "risk-free rate=5%. "
        "Step 1: Solve for equity value E and equity volatility sigma_E using Merton equations iteratively (Newton's method, 20 iterations). "
        "Step 2: Compute distance-to-default (DD) and probability of default (PD). "
        "Step 3: Compute credit spread on the debt. "
        "Step 4: Run sensitivity analysis — recompute PD for asset values [120,130,140,150,160,170]M. "
        "Print all values including intermediate Newton iterations.",
        "finance", ["merton_model", "credit_risk", "distance_to_default", "newton"], domain="finance",
    ),
    BenchTask(
        "Implement a complete options Greeks calculator and delta-hedging simulation: "
        "Use Black-Scholes for European call: S=100, K=100, T=30/365, r=0.05, sigma=0.25. "
        "Compute all 5 Greeks: Delta, Gamma, Theta, Vega, Rho. "
        "Then simulate delta-hedging over 30 days (seed=42, daily rebalancing): "
        "stock follows GBM, rebalance delta daily, track hedging P&L vs unhedged. "
        "Compute hedge effectiveness ratio. "
        "Also price an Asian option (arithmetic average, same params) via Monte Carlo (10000 paths). "
        "Print all Greeks, daily hedge P&L, and Asian option price.",
        "finance", ["black_scholes", "greeks", "delta_hedging", "asian_option"], domain="finance",
    ),
    BenchTask(
        "Build a leveraged buyout (LBO) model: "
        "Acquisition price=$500M (8x EBITDA=$62.5M), financed 60% debt at 8% interest. "
        "Revenue=$250M growing 10%/yr, EBITDA margin expands from 25% to 32% over 5 years. "
        "D&A=$15M/yr, CapEx=4% of revenue, working capital=8% of revenue change. "
        "Debt amortizes 5%/yr of original principal, remainder bullet at exit. "
        "Exit at year 5 at 9x EBITDA. "
        "Build full 5-year P&L, FCF, and debt schedule. "
        "Compute equity IRR, MOIC, and cash-on-cash return. "
        "Show returns at exit multiples of 7x, 8x, 9x, 10x.",
        "finance", ["lbo", "irr", "moic", "debt_schedule", "fcf"], domain="finance",
    ),
    BenchTask(
        "Implement a full yield curve construction and interpolation engine: "
        "Bootstrap a zero curve from these par swap rates: "
        "6m=4.5%, 1y=4.8%, 2y=5.1%, 3y=5.3%, 5y=5.6%, 7y=5.75%, 10y=5.9%. "
        "Step 1: Bootstrap zero rates for each maturity using exact bootstrapping. "
        "Step 2: Implement cubic spline interpolation to get zero rates at 0.25y intervals up to 10y. "
        "Step 3: Convert zero rates to discount factors and forward rates. "
        "Step 4: Price a 4-year interest rate swap (fixed=5.4%, notional=$10M, semi-annual) using the curve. "
        "Print zero curve, forward curve, and swap NPV.",
        "finance", ["yield_curve", "bootstrapping", "cubic_spline", "swap_pricing"], domain="finance",
    ),
    BenchTask(
        "Build a momentum + mean-reversion regime-switching trading system: "
        "Generate 504 trading days (2 years) of prices using GBM (seed=99, mu=0.07, sigma=0.18, S0=100). "
        "Step 1: Identify regimes using 20-day realized volatility — high vol (>20% ann.) vs low vol. "
        "Step 2: In low-vol regime use momentum (buy if 20d return > 2%, sell if < -2%). "
        "Step 3: In high-vol regime use mean-reversion (buy if 5d return < -3%, sell if > 3%). "
        "Step 4: Apply 0.1% transaction cost per trade. "
        "Step 5: Compare strategy vs buy-and-hold: annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar ratio. "
        "Print regime classification for each month and full performance attribution.",
        "finance", ["regime_switching", "momentum", "mean_reversion", "sharpe", "sortino", "calmar"], domain="finance",
    ),
]

REAL_ESTATE_TASKS: list[BenchTask] = [
    BenchTask(
        "Model a full private equity real estate waterfall distribution: "
        "LP commits $8M (90%), GP commits $0.9M (10%). Total equity=$8.9M. "
        "Preferred return=8% cumulative (not compounded). "
        "Waterfall tiers: (1) return of capital to all, (2) 8% pref to LPs, "
        "(3) GP catch-up to 20% of total profits, (4) 80/20 LP/GP split on remaining. "
        "Investment holds 4 years. Cash flows: year1=$500k, year2=$700k, year3=$800k, year4=$12M (sale). "
        "Compute year-by-year distribution to LP and GP, IRR for each, equity multiple. "
        "Show full waterfall table.",
        "real_estate", ["waterfall", "preferred_return", "gp_catch_up", "irr", "equity_multiple"], domain="real_estate",
    ),
    BenchTask(
        "Build a complete real estate development pro forma: "
        "Land cost=$2M, hard costs=$8M (12-month construction), soft costs=15% of hard, "
        "financing costs: construction loan=$8M at 9% interest-only (drawn evenly over 12 months). "
        "Stabilized NOI=$1.1M (85% occupancy, gross rents=$1.5M, expenses=25%). "
        "Permanent financing: 65% LTV at 6.5% / 30yr amortization. "
        "Exit cap rate=5.5% at end of year 2 post-completion. "
        "Compute: total project cost, development yield, exit value, net profit, "
        "equity multiple on $2.9M equity, project IRR. "
        "Show full sources & uses, construction draw schedule, and return summary.",
        "real_estate", ["development_proforma", "construction_loan", "development_yield", "irr"], domain="real_estate",
    ),
    BenchTask(
        "Perform a full CMBS loan sizing and underwriting analysis: "
        "Property: 200-unit multifamily, market rents=$1,800/unit/month, vacancy=7%, "
        "other income=$50k/yr, operating expenses=$2,200/unit/yr, reserves=$300/unit/yr. "
        "Lender requirements: max LTV=65%, min DSCR=1.25x, debt yield>8.5%. "
        "Available loan terms: 10yr fixed at 6.75%, 30yr amortization. "
        "Step 1: Calculate EGI, NOI, and property value at 5.0% cap rate. "
        "Step 2: Size loan based on all 3 constraints — which is binding? "
        "Step 3: Compute actual DSCR, LTV, debt yield on the sized loan. "
        "Step 4: Stress test — what happens at 15% vacancy and 10% expense increase? "
        "Print full underwriting sheet.",
        "real_estate", ["cmbs", "dscr", "debt_yield", "loan_sizing", "stress_test"], domain="real_estate",
    ),
    BenchTask(
        "Build a complete 10-year hold period real estate portfolio IRR model: "
        "Acquire 3 properties simultaneously: "
        "Office: $5M at 6.5% cap, NOI grows 2%/yr, sell yr10 at 7% cap. "
        "Retail: $3M at 7% cap, NOI grows 1.5%/yr (lease rollover risk: yr5 NOI drops 15%), sell yr10 at 7.5% cap. "
        "Industrial: $4M at 5.5% cap, NOI grows 3%/yr, sell yr7 at 5% cap and reinvest in bonds at 5.5%. "
        "All properties: 60% LTV at 6% / 25yr amortization. "
        "Portfolio-level: 2% acquisition cost, 2% disposition cost. "
        "Compute: individual property IRR and equity multiple, portfolio blended IRR, "
        "attribution by property, and sensitivity to cap rate expansion (+50bps, +100bps).",
        "real_estate", ["portfolio_irr", "hold_period", "cap_rate_sensitivity", "attribution"], domain="real_estate",
    ),
    BenchTask(
        "Model a complex triple-net (NNN) lease vs gross lease comparison with NPV analysis: "
        "Property value=$4M, 10yr lease. "
        "NNN option: base rent=$200k/yr (2% annual escalation), tenant pays all expenses ($80k/yr). "
        "Gross option: base rent=$270k/yr (fixed), landlord pays all expenses ($80k/yr, growing 3%/yr). "
        "Financing: 65% LTV at 6.5%, 25yr amortization. "
        "Tax: depreciation over 39 years (commercial), 25% tax bracket, passive loss rules apply. "
        "Compute for each lease structure: annual NOI, after-tax cash flow, NPV at 8% discount rate, "
        "after-tax IRR assuming 5.5% cap exit. "
        "Also compute breakeven rent where NPV is equal. Print full 10-year cash flow table for each.",
        "real_estate", ["nnn_lease", "gross_lease", "npv", "after_tax_irr", "depreciation"], domain="real_estate",
    ),
    BenchTask(
        "Build a real estate opportunity zone (OZ) investment model with tax deferral: "
        "Investor has $500k capital gain. Invests in OZ fund on day 1. "
        "OZ rules: (1) original gain deferred until 2026 (5 years), (2) 10% step-up in basis after 5yr hold, "
        "(3) appreciation in OZ fund is tax-free after 10yr hold. "
        "OZ fund: invests in development project, equity=$500k + $1M debt at 7%. "
        "NOI grows from $90k to $130k over 10 years (linear). Exit at 5.5% cap in yr10. "
        "Compare 3 scenarios: (A) OZ investment, (B) same deal without OZ, (C) invest gain in S&P500 (7% annual). "
        "For each: compute after-tax IRR assuming 23.8% capital gains rate and 37% ordinary income. "
        "Print full cash flow model and tax savings from OZ treatment.",
        "real_estate", ["opportunity_zone", "tax_deferral", "after_tax_irr", "basis_step_up"], domain="real_estate",
    ),
    BenchTask(
        "Implement a full real estate Monte Carlo simulation for risk analysis: "
        "Apartment building: purchase=$6M, 65% LTV at 6.5%/30yr, NOI=$330k at stabilization. "
        "Stochastic inputs (seed=42, 10,000 simulations): "
        "rent growth: normal(2%, 3%) annually, vacancy: uniform(5%,12%), "
        "expense growth: normal(3%, 1.5%), exit cap rate: normal(5.5%, 0.75%). "
        "Hold period=5 years. Compute: distribution of 5yr IRR, equity multiple, "
        "probability of loss (IRR<0%), probability of achieving 12% IRR, "
        "5th/25th/50th/75th/95th percentile outcomes, expected shortfall below 5th pct. "
        "Plot histogram description and print full risk metrics table.",
        "real_estate", ["monte_carlo", "irr_distribution", "risk_metrics", "stochastic"], domain="real_estate",
    ),
    BenchTask(
        "Model a 1031 exchange with reverse exchange and boot calculation: "
        "Relinquished property: sold for $1.2M, original basis=$300k, accumulated depreciation=$150k. "
        "Tax rates: depreciation recapture=25%, capital gains=20%, NIIT=3.8%. "
        "Replacement property A: $1.5M (all-cash, no boot). "
        "Replacement property B: $1M + $100k cash received back (cash boot). "
        "Replacement property C: $1.3M with $200k mortgage assumed by buyer + $100k mortgage on replacement. "
        "For each scenario compute: recognized gain, deferred gain, tax liability, net benefit of exchange vs sale. "
        "Also compute: new adjusted basis in each replacement property. "
        "Print full gain recognition and tax analysis for all 3 scenarios.",
        "real_estate", ["1031_exchange", "boot", "depreciation_recapture", "basis", "deferred_gain"], domain="real_estate",
    ),
    BenchTask(
        "Build a ground lease vs fee simple ownership NPV analysis: "
        "Property: office building, land value=$2M, improvements=$8M. "
        "Fee simple: NOI=$650k/yr, growing 2.5%/yr, sell at 6% cap in yr15, 65% LTV at 6.75%. "
        "Ground lease alternative: land lease=$80k/yr (fixed 15yr), improvements financed 70% LTV at 6.5%. "
        "Leaseholder NOI=$650k minus land rent. After 15yr: lease resets to 8% of then-land-value (appraise land at 3%/yr growth). "
        "For both: build 20-year cash flow model, compute levered IRR, unlevered IRR, NPV at 8% discount. "
        "Compute breakeven land lease rate where both structures have equal NPV. "
        "Print full comparison table.",
        "real_estate", ["ground_lease", "fee_simple", "npv", "irr", "lease_reset"], domain="real_estate",
    ),
    BenchTask(
        "Implement a complete real estate portfolio stress test (bank-style): "
        "Portfolio of 5 properties (all acquired at 65% LTV, 25yr amortization, 6% rate): "
        "Property 1: Multifamily $5M, cap=5.5%, NOI=$275k. "
        "Property 2: Office $8M, cap=6.5%, NOI=$520k. "
        "Property 3: Retail $3M, cap=7%, NOI=$210k. "
        "Property 4: Industrial $6M, cap=5%, NOI=$300k. "
        "Property 5: Hotel $10M, NOI=$700k (RevPAR-driven, high vol). "
        "Stress scenarios: (A) base, (B) rents -15% across all, (C) cap rates +150bps, "
        "(D) vacancy +10ppts, (E) combined severe (rents -20%, caps +200bps, vacancy +15ppts). "
        "For each scenario: compute portfolio NOI, DSCR, LTV (revalued), equity value, properties in distress (DSCR<1.0). "
        "Identify which properties breach covenants first. Print full stress test matrix.",
        "real_estate", ["stress_test", "portfolio", "dscr", "covenant_breach", "ltv_revaluation"], domain="real_estate",
    ),
]

# Domain repeat tasks — structurally similar, test memory transfer
FINANCE_REPEAT_TASKS: list[BenchTask] = [
    BenchTask(
        "Build a leveraged recapitalization model (reverse LBO): "
        "Public company: EBITDA=$80M, current debt=$50M, equity market cap=$600M. "
        "Recapitalize by adding $300M debt at 7.5% to pay special dividend. "
        "Post-recap: model 5-year P&L (EBITDA grows 8%/yr, D&A=$20M, CapEx=$25M, taxes=28%). "
        "Compute: interest coverage ratio each year, FCF available for deleveraging, "
        "debt paydown schedule, credit metrics (Net Debt/EBITDA). "
        "At what year does leverage return to pre-recap level? Print full model.",
        "finance", ["leveraged_recap", "credit_metrics", "deleveraging", "fcf"], domain="finance",
    ),
    BenchTask(
        "Implement a convertible bond pricing model: "
        "Bond: face=$1000, coupon=3%, maturity=5yr, conversion ratio=20 shares (conversion price=$50). "
        "Stock: current price=$42, volatility=35%, risk-free=5%. "
        "Step 1: Price straight bond component (YTM=7% for comparable non-convertible). "
        "Step 2: Price conversion option using Black-Scholes (European, no dividends). "
        "Step 3: Compute convertible bond value = bond floor + option value. "
        "Step 4: Compute delta, gamma of the convertible. "
        "Step 5: Calculate breakeven premium and payback period. "
        "Print all components and sensitivity table (stock price from $30 to $70 in $5 steps).",
        "finance", ["convertible_bond", "bond_floor", "conversion_option", "breakeven"], domain="finance",
    ),
    BenchTask(
        "Build a complete risk parity portfolio with leverage: "
        "4 asset classes: equities(vol=18%), bonds(vol=7%), commodities(vol=22%), REITs(vol=15%). "
        "Correlation matrix: eq-bond=-0.2, eq-comm=0.1, eq-reit=0.6, bond-comm=-0.1, bond-reit=-0.1, comm-reit=0.2. "
        "Step 1: Find risk parity weights (each asset contributes equally to portfolio risk). "
        "Step 2: Leverage portfolio to 10% target volatility. "
        "Step 3: Compare vs 60/40 portfolio (expected returns: eq=9%, bond=4%, comm=7%, reit=8%). "
        "Step 4: Compute marginal risk contribution and component VaR for each asset. "
        "Print weights, leverage ratio, expected return, Sharpe (rf=4%), and risk attribution.",
        "finance", ["risk_parity", "leverage", "marginal_risk", "component_var"], domain="finance",
    ),
]

REAL_ESTATE_REPEAT_TASKS: list[BenchTask] = [
    BenchTask(
        "Model a sale-leaseback transaction with present value analysis: "
        "Company owns headquarters: current book value=$3M, fair market value=$8M. "
        "Sale-leaseback terms: sell for $8M, lease back for 20 years at $520k/yr (2% annual escalation). "
        "Buyer's target return: 6.5% cap rate at acquisition, requires 8% IRR. "
        "Company's cost of capital: 9% (WACC). "
        "Compute: (1) gain on sale and tax impact (25% rate), (2) net proceeds after tax, "
        "(3) present value of lease obligations (ASC 842 right-of-use asset), "
        "(4) NPV of transaction from company's perspective vs keeping property with 3% appreciation. "
        "(5) Breakeven lease rate where transaction NPV=0. Print full analysis.",
        "real_estate", ["sale_leaseback", "right_of_use", "npv", "breakeven_rent"], domain="real_estate",
    ),
    BenchTask(
        "Build a real estate syndication waterfall with preferred equity and mezz debt: "
        "Capital stack: senior debt=$5M at 6% (IO), mezz debt=$1.5M at 12%, "
        "preferred equity=$1M at 10% cumulative, common equity=$2M (60% LP / 40% GP). "
        "NOI year 1-3: $480k/yr, year 4-5: $520k/yr. Exit year 5 at 5.75% cap. "
        "Distribution priority: (1) debt service, (2) mezz interest, (3) pref return+principal, "
        "(4) common equity pref return 8%, (5) GP catch-up 50% to 20% of profits, (6) 80/20 LP/GP. "
        "Compute: year-by-year distributions to each tranche, IRR per tranche, "
        "equity multiple for LP and GP common. Print full capital stack waterfall.",
        "real_estate", ["capital_stack", "mezz_debt", "preferred_equity", "waterfall", "irr"], domain="real_estate",
    ),
]

ALL_DOMAIN_TASKS: list[BenchTask] = FINANCE_TASKS + REAL_ESTATE_TASKS
ALL_DOMAIN_REPEAT_TASKS: list[BenchTask] = FINANCE_REPEAT_TASKS + REAL_ESTATE_REPEAT_TASKS
