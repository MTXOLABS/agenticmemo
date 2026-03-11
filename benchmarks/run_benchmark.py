"""AgenticMemo Benchmark Runner
================================

Three modes:

  --research   (Recommended for research paper)
               Runs 5 configurations side-by-side:
                 • Claude Standalone   (raw Claude, no AgenticMemo)
                 • Claude + AgenticMemo
                 • GPT Standalone      (raw GPT, no AgenticMemo)
                 • GPT + AgenticMemo
                 • Mock + AgenticMemo  (deterministic baseline)
               Produces a clear Δ-improvement table for every suite.

  --compare    Quick comparison of available providers (all with AgenticMemo).

  (default)    Single-provider run. Use --provider to pick one.

Usage
-----
    # Research paper comparison (needs both API keys):
    python -m benchmarks.run_benchmark --research

    # Run just one suite:
    python -m benchmarks.run_benchmark --research --suite core

    # Quick compare (all providers, AgenticMemo enabled):
    python -m benchmarks.run_benchmark --compare

    # Single provider:
    python -m benchmarks.run_benchmark --provider anthropic --suite core

    # Save JSON:
    python -m benchmarks.run_benchmark --research --output paper_results.json

API keys: ANTHROPIC_API_KEY and OPENAI_API_KEY (env vars or --anthropic-key / --openai-key)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich import box

console = Console()

sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticmemo import Agent, AgentConfig, LearningConfig, MemoryConfig, RetrievalConfig
from agenticmemo.memory.case import Case, CaseOutcome
from agenticmemo.memory.shared import SharedMemoryPool
from agenticmemo.retrieval.ensemble import EnsembleRetriever
from agenticmemo.tools import PythonReplTool
from agenticmemo.types import TaskStatus, Trajectory

from .mock_llm import MockLLM
from .tasks import (
    ALL_REPEAT_TASKS,
    ALL_SOLO_TASKS,
    ALL_HARD_TASKS,
    ALL_HARD_REPEAT_TASKS,
    HARD_FAILURE_TASKS,
    FAILURE_TASKS,
    MULTIAGENT_TASKS,
    BenchTask,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration identifiers
# ─────────────────────────────────────────────────────────────────────────────

# Each config is (label, provider, agenticmemo_enabled)
# No mock — research mode uses only real LLM providers
RESEARCH_CONFIGS: list[tuple[str, str, bool]] = [
    ("Claude Standalone",    "anthropic", False),
    ("Claude + AgenticMemo", "anthropic", True),
    ("GPT Standalone",       "openai",    False),
    ("GPT + AgenticMemo",    "openai",    True),
]

_CONFIG_COLORS = {
    "Claude Standalone":    "red",
    "Claude + AgenticMemo": "magenta",
    "GPT Standalone":       "yellow",
    "GPT + AgenticMemo":    "blue",
}

_PROVIDER_COLORS = {
    "mock":      "cyan",
    "anthropic": "magenta",
    "openai":    "blue",
}

_PROVIDER_LABELS = {
    "mock":      "AgenticMemo (Mock)",
    "anthropic": "Anthropic (Claude)",
    "openai":    "OpenAI (GPT)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task: str
    status: str
    steps: int
    tokens: int
    duration_ms: float
    reward: float
    reflexion_triggered: bool = False


@dataclass
class SuiteResult:
    name: str
    config_label: str = "unknown"
    tasks_total: int = 0
    tasks_success: int = 0
    tasks_partial: int = 0
    tasks_failed: int = 0
    avg_steps: float = 0.0
    avg_tokens: float = 0.0
    avg_duration_ms: float = 0.0
    avg_reward: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.tasks_success / self.tasks_total if self.tasks_total else 0.0

    def finish(self) -> None:
        if not self.task_results:
            return
        self.tasks_total   = len(self.task_results)
        self.tasks_success = sum(1 for r in self.task_results if r.status == "success")
        self.tasks_partial = sum(1 for r in self.task_results if r.status == "partial")
        self.tasks_failed  = sum(1 for r in self.task_results if r.status == "failure")
        self.avg_steps     = sum(r.steps for r in self.task_results) / self.tasks_total
        self.avg_tokens    = sum(r.tokens for r in self.task_results) / self.tasks_total
        self.avg_duration_ms = sum(r.duration_ms for r in self.task_results) / self.tasks_total
        self.avg_reward    = sum(r.reward for r in self.task_results) / self.tasks_total


# ─────────────────────────────────────────────────────────────────────────────
# Agent factories
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(provider: str, api_key: str | None, model: str | None):
    if provider == "anthropic":
        from agenticmemo.llm.anthropic_llm import AnthropicLLM  # noqa: PLC0415
        return AnthropicLLM(model=model or "claude-sonnet-4-6", api_key=api_key)
    if provider == "openai":
        from agenticmemo.llm.openai_llm import OpenAILLM  # noqa: PLC0415
        return OpenAILLM(model=model or "gpt-5-mini-2025-08-07", api_key=api_key)
    return MockLLM(fail_rate=0.0)


def make_standalone_agent(
    provider: str,
    api_key: str | None,
    model: str | None = None,
) -> Agent:
    """Plain LLM agent — NO memory, NO hints, NO GRPO, NO reflexion."""
    cfg = AgentConfig(
        verbose=False,
        max_steps=10,
        max_retries=1,
        memory=MemoryConfig(
            min_reward_to_store=999.0,   # effectively store nothing
            max_cases=0,
        ),
        retrieval=RetrievalConfig(
            top_k=0,                     # retrieve nothing
            enable_grpo=False,
        ),
        learning=LearningConfig(
            enable_reflexion=False,
            enable_quality_filter=False,
        ),
    )
    llm = _build_llm(provider, api_key, model)
    agent = Agent(llm, cfg)
    agent.add_tool(PythonReplTool())
    return agent


def make_agenticmemo_agent(
    provider: str,
    api_key: str | None,
    model: str | None = None,
    fail_rate: float = 0.0,
) -> Agent:
    """Full AgenticMemo stack — memory + GRPO + reflexion + hints."""
    if provider == "mock":
        llm = MockLLM(fail_rate=fail_rate)
    else:
        llm = _build_llm(provider, api_key, model)

    cfg = AgentConfig(
        verbose=False,
        max_steps=10,
        max_retries=2,
        memory=MemoryConfig(min_reward_to_store=-1.0),
        retrieval=RetrievalConfig(top_k=3, enable_grpo=True, grpo_update_every=3),
        learning=LearningConfig(
            enable_reflexion=True,
            max_reflexion_retries=1,
            enable_quality_filter=True,
            success_reward=1.0,
            partial_reward=0.3,
            failure_reward=-0.2,
        ),
    )
    agent = Agent(llm, cfg)
    agent.add_tool(PythonReplTool())
    return agent


# Legacy factory for --compare / --provider modes
def make_agent(
    provider: str = "mock",
    api_key: str | None = None,
    model: str | None = None,
    fail_rate: float = 0.0,
    enable_reflexion: bool = True,
    enable_grpo: bool = True,
    enable_filter: bool = True,
    persist_path: str | None = None,
) -> Agent:
    cfg = AgentConfig(
        verbose=False,
        max_steps=10,
        max_retries=2,
        memory=MemoryConfig(persist_path=persist_path, min_reward_to_store=-1.0),
        retrieval=RetrievalConfig(top_k=3, enable_grpo=enable_grpo, grpo_update_every=3),
        learning=LearningConfig(
            enable_reflexion=enable_reflexion,
            max_reflexion_retries=1,
            enable_quality_filter=enable_filter,
            success_reward=1.0,
            partial_reward=0.3,
            failure_reward=-0.2,
        ),
    )
    if provider == "anthropic":
        agent = Agent.from_anthropic(api_key=api_key, model=model or "claude-sonnet-4-6", cfg=cfg)
    elif provider == "openai":
        agent = Agent.from_openai(api_key=api_key, model=model or "gpt-4o", cfg=cfg)
    else:
        agent = Agent(MockLLM(fail_rate=fail_rate), cfg)
    agent.add_tool(PythonReplTool())
    return agent


async def run_task(agent: Agent, bench: BenchTask) -> TaskResult:
    t0 = time.time()
    traj = await agent.run(bench.task)
    duration = (time.time() - t0) * 1000
    from agenticmemo.learning.filters import TrajectoryFilter  # noqa: PLC0415
    reward = TrajectoryFilter().assign_reward(traj)
    return TaskResult(
        task=bench.task[:70],
        status=traj.status.value,
        steps=traj.num_steps,
        tokens=traj.total_tokens,
        duration_ms=duration,
        reward=reward,
        reflexion_triggered=bool(traj.reflection),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Research suites — each returns one SuiteResult per config
# ─────────────────────────────────────────────────────────────────────────────

async def research_core(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Suite 1: Core accuracy on first-time tasks (no prior memory)."""
    result = SuiteResult(name="Core Accuracy", config_label=config_label)
    agent = (
        make_agenticmemo_agent(provider, api_key, model)
        if agenticmemo else
        make_standalone_agent(provider, api_key, model)
    )
    with Progress(SpinnerColumn(), TextColumn(f"[dim][{config_label}] {{task.description}}"),
                  BarColumn(), TaskProgressColumn(), console=console, transient=True) as prog:
        pt = prog.add_task("tasks...", total=len(ALL_SOLO_TASKS))
        for bench in ALL_SOLO_TASKS:
            r = await run_task(agent, bench)
            result.task_results.append(r)
            prog.advance(pt)
    result.finish()
    if agenticmemo:
        result.extra["memory_size"] = await agent.memory_size()
    return result


async def research_memory_transfer(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Suite 2: Memory transfer — warmup then repeat tasks.

    Standalone re-runs each task cold.
    AgenticMemo retrieves relevant experience from warmup → should do better.
    """
    result = SuiteResult(name="Memory Transfer", config_label=config_label)
    agent = (
        make_agenticmemo_agent(provider, api_key, model)
        if agenticmemo else
        make_standalone_agent(provider, api_key, model)
    )

    # Warmup phase — both agents run these (AgenticMemo stores them, Standalone ignores)
    console.print(f"  [dim][{config_label}] Phase A: Warmup ({len(ALL_SOLO_TASKS)} tasks)...[/]")
    for bench in ALL_SOLO_TASKS:
        await run_task(agent, bench)

    warmup_cases = await agent.memory_size() if agenticmemo else 0

    # Test phase — measure improvement on similar repeat tasks
    console.print(f"  [dim][{config_label}] Phase B: Repeat tasks (measuring transfer)...[/]")
    for bench in ALL_REPEAT_TASKS:
        r = await run_task(agent, bench)
        result.task_results.append(r)

    result.finish()
    result.extra["warmup_cases_in_memory"] = warmup_cases
    return result


async def research_reflexion(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Suite 3: Failure recovery.

    Standalone has no reflexion — fails and stays failed.
    AgenticMemo diagnoses failures, corrects, and retries.
    Uses fail_rate=0.7 for Mock; real LLMs rely on natural failures.
    """
    result = SuiteResult(name="Reflexion Recovery", config_label=config_label)

    fail_rate = 0.7 if provider == "mock" else 0.0
    if agenticmemo:
        agent = make_agenticmemo_agent(provider, api_key, model, fail_rate=fail_rate)
        agent_no_ref = make_agenticmemo_agent(provider, api_key, model, fail_rate=fail_rate)
        # Disable reflexion on the second one for fair comparison
        agent_no_ref._cfg.learning.enable_reflexion = False
    else:
        # Standalone: no reflexion either way
        agent = make_standalone_agent(provider, api_key, model)
        agent_no_ref = None

    tasks = FAILURE_TASKS + ALL_SOLO_TASKS[:4]

    console.print(f"  [dim][{config_label}] Running {len(tasks)} tasks...[/]")
    for bench in tasks:
        r = await run_task(agent, bench)
        result.task_results.append(r)

    result.finish()
    result.extra["reflexion_enabled"] = agenticmemo
    result.extra["reflexion_triggered"] = sum(1 for r in result.task_results if r.reflexion_triggered)
    return result


async def research_efficiency(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Suite 4: Token/step efficiency after learning.

    After warmup, AgenticMemo agents receive experience context that helps
    them solve tasks in fewer steps and tokens.
    """
    result = SuiteResult(name="Efficiency (Steps & Tokens)", config_label=config_label)
    agent = (
        make_agenticmemo_agent(provider, api_key, model)
        if agenticmemo else
        make_standalone_agent(provider, api_key, model)
    )

    # Warmup (AgenticMemo builds memory, Standalone ignores)
    if agenticmemo:
        console.print(f"  [dim][{config_label}] Warmup phase...[/]")
        for bench in ALL_SOLO_TASKS[:6]:
            await run_task(agent, bench)

    # Measurement phase
    console.print(f"  [dim][{config_label}] Measuring efficiency...[/]")
    for bench in ALL_REPEAT_TASKS + ALL_SOLO_TASKS[6:]:
        r = await run_task(agent, bench)
        result.task_results.append(r)

    result.finish()
    return result


async def research_retrieval(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Suite 5: Retrieval quality — ensemble vs semantic-only (AgenticMemo only).

    Standalone has no retrieval so it scores 0 by definition.
    """
    from agenticmemo.config import RetrievalConfig as RC  # noqa: PLC0415

    result = SuiteResult(name="Retrieval Quality", config_label=config_label)

    if not agenticmemo:
        result.tasks_total = 6
        result.extra = {
            "note": "N/A — standalone agent has no retrieval",
            "ensemble_precision_at_3": 0.0,
            "semantic_precision_at_3": 0.0,
        }
        return result

    agent = make_agenticmemo_agent(provider, api_key, model)
    for bench in ALL_SOLO_TASKS:
        await run_task(agent, bench)

    mem = agent._memory
    cases = await mem.all_cases()

    queries = [
        ("fibonacci sequence algorithm",  "fibonacci"),
        ("check if a number is divisible","prime"),
        ("reverse a string characters",   "reverse"),
        ("stack data structure push pop",  "stack"),
        ("binary search sorted array",    "binary_search"),
        ("string anagram checker",        "anagram"),
    ]

    ens_r = EnsembleRetriever(mem, RC(weight_semantic=0.5, weight_bm25=0.3,
                                      weight_graph=0.1, weight_temporal=0.1, top_k=3))
    sem_r = EnsembleRetriever(mem, RC(weight_semantic=1.0, weight_bm25=0.0,
                                      weight_graph=0.0, weight_temporal=0.0, top_k=3))
    await ens_r.rebuild_index()
    await sem_r.rebuild_index()

    ens_hits = sem_hits = 0
    console.print(f"  [dim][{config_label}] Evaluating {len(queries)} queries over {len(cases)} cases...[/]")
    for query, kw in queries:
        if any(kw in c.task.lower() for c, _ in await ens_r.retrieve(query, top_k=3)):
            ens_hits += 1
        if any(kw in c.task.lower() for c, _ in await sem_r.retrieve(query, top_k=3)):
            sem_hits += 1

    total = len(queries)
    result.tasks_total   = total
    result.tasks_success = ens_hits
    result.extra = {
        "corpus_size": len(cases),
        "ensemble_precision_at_3": round(ens_hits / total, 3),
        "semantic_precision_at_3": round(sem_hits / total, 3),
        "ensemble_vs_semantic_improvement_pct": round(
            (ens_hits - sem_hits) / max(sem_hits, 1) * 100, 1
        ),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Research suite registry
# ─────────────────────────────────────────────────────────────────────────────

async def research_hard_core(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Hard Suite 1: 18 algorithm/system-design tasks — cold start."""
    result = SuiteResult(name="Hard Core Accuracy", config_label=config_label)
    agent = (
        make_agenticmemo_agent(provider, api_key, model)
        if agenticmemo else
        make_standalone_agent(provider, api_key, model)
    )
    tasks = ALL_HARD_TASKS
    with Progress(SpinnerColumn(), TextColumn(f"[dim][{config_label}] {{task.description}}"),
                  BarColumn(), TaskProgressColumn(), console=console, transient=True) as prog:
        pt = prog.add_task("hard tasks...", total=len(tasks))
        for bench in tasks:
            r = await run_task(agent, bench)
            result.task_results.append(r)
            prog.advance(pt)
    result.finish()
    if agenticmemo:
        result.extra["memory_size"] = await agent.memory_size()
    return result


async def research_hard_transfer(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Hard Suite 2: Memory transfer — warmup on 18 hard tasks, then 5 structurally
    similar but lexically distinct tasks (LFU after LRU, Bellman-Ford after Dijkstra, etc.)
    """
    result = SuiteResult(name="Hard Memory Transfer", config_label=config_label)
    agent = (
        make_agenticmemo_agent(provider, api_key, model)
        if agenticmemo else
        make_standalone_agent(provider, api_key, model)
    )

    console.print(f"  [dim][{config_label}] Warmup: {len(ALL_HARD_TASKS)} hard tasks...[/]")
    for bench in ALL_HARD_TASKS:
        await run_task(agent, bench)

    warmup_cases = await agent.memory_size() if agenticmemo else 0

    console.print(f"  [dim][{config_label}] Transfer phase: {len(ALL_HARD_REPEAT_TASKS)} tasks...[/]")
    for bench in ALL_HARD_REPEAT_TASKS:
        r = await run_task(agent, bench)
        result.task_results.append(r)

    result.finish()
    result.extra["warmup_cases_in_memory"] = warmup_cases
    return result


async def research_hard_reflexion(
    config_label: str, provider: str, api_key: str | None,
    agenticmemo: bool, model: str | None,
) -> SuiteResult:
    """Hard Suite 3: Failure recovery on 3 deliberately tricky tasks
    (regex matching DP, trapping rain water, thread-safe queue).
    """
    result = SuiteResult(name="Hard Reflexion Recovery", config_label=config_label)

    if agenticmemo:
        agent = make_agenticmemo_agent(provider, api_key, model)
    else:
        agent = make_standalone_agent(provider, api_key, model)

    tasks = HARD_FAILURE_TASKS

    console.print(f"  [dim][{config_label}] {len(tasks)} hard failure tasks...[/]")
    for bench in tasks:
        r = await run_task(agent, bench)
        result.task_results.append(r)

    result.finish()
    result.extra["reflexion_enabled"] = agenticmemo
    result.extra["reflexion_triggered"] = sum(1 for r in result.task_results if r.reflexion_triggered)
    return result


RESEARCH_SUITE_MAP = {
    "core":         research_core,
    "memory":       research_memory_transfer,
    "reflexion":    research_reflexion,
    "efficiency":   research_efficiency,
    "retrieval":    research_retrieval,
    "hard_core":    research_hard_core,
    "hard_transfer":research_hard_transfer,
    "hard_reflexion":research_hard_reflexion,
}

# Legacy suites for --compare / --provider modes
async def bench_core(provider, api_key, model=None):
    return await research_core("run", provider, api_key, True, model)

async def bench_memory(provider, api_key, model=None):
    return await research_memory_transfer("run", provider, api_key, True, model)

async def bench_retrieval(provider, api_key, model=None):
    return await research_retrieval("run", provider, api_key, True, model)

async def bench_reflexion(provider, api_key, model=None):
    return await research_reflexion("run", provider, api_key, True, model)

async def bench_efficiency(provider, api_key, model=None):
    return await research_efficiency("run", provider, api_key, True, model)

async def _bench_wrap(suite_fn, provider, api_key, model, agenticmemo=True):
    """Wrap a research suite function for use in single/compare mode."""
    return await suite_fn("run", provider, api_key, agenticmemo, model)

async def bench_hard_core(provider, api_key, model=None):
    return await research_hard_core("run", provider, api_key, True, model)

async def bench_hard_transfer(provider, api_key, model=None):
    return await research_hard_transfer("run", provider, api_key, True, model)

async def bench_hard_reflexion(provider, api_key, model=None):
    return await research_hard_reflexion("run", provider, api_key, True, model)

SUITE_MAP = {
    "core":           bench_core,
    "memory":         bench_memory,
    "retrieval":      bench_retrieval,
    "reflexion":      bench_reflexion,
    "efficiency":     bench_efficiency,
    "hard_core":      bench_hard_core,
    "hard_transfer":  bench_hard_transfer,
    "hard_reflexion": bench_hard_reflexion,
}


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _color(rate: float) -> str:
    if rate >= 0.85: return "green"
    if rate >= 0.60: return "yellow"
    return "red"


def _delta_str(a: float, b: float, pct: bool = True) -> str:
    """Return a colored Δ string (a = with_memo, b = standalone)."""
    diff = a - b
    if pct:
        diff_str = f"{diff*100:+.1f}pp"
    else:
        diff_str = f"{diff:+.2f}"
    color = "green" if diff > 0 else "red" if diff < 0 else "dim"
    return f"[{color}]{diff_str}[/]"


def print_suite_result(r: SuiteResult) -> None:
    color = _color(r.success_rate)
    clabel = r.config_label
    ccolor = _CONFIG_COLORS.get(clabel, "white")
    console.print(f"\n  [{ccolor}]{clabel}[/]  →  {r.name}")

    if r.tasks_total > 0:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column(style="dim", min_width=20)
        t.add_column(justify="right")
        t.add_row("Tasks",        str(r.tasks_total))
        t.add_row("Success",      f"[{color}]{r.tasks_success} ({r.success_rate:.1%})[/]")
        t.add_row("Avg steps",    f"{r.avg_steps:.1f}")
        t.add_row("Avg tokens",   f"{r.avg_tokens:.0f}")
        t.add_row("Avg duration", f"{r.avg_duration_ms:.0f} ms")
        t.add_row("Avg reward",   f"{r.avg_reward:.3f}")
        console.print(t)

    if r.extra:
        for k, v in r.extra.items():
            if k == "note":
                console.print(f"  [dim]{v}[/]")
            elif not isinstance(v, (list, dict)):
                console.print(f"  [dim]{k.replace('_',' ').title()}:[/] {v}")


def print_research_comparison(
    results: dict[str, dict[str, SuiteResult]],
    suite_names: list[str],
) -> None:
    """Print the core research comparison table."""
    console.print("\n")
    console.rule("[bold green]Research Comparison: Standalone vs AgenticMemo")

    for suite_name in suite_names:
        suite_display = suite_name.replace("_", " ").title()
        console.print(f"\n[bold white]Suite: {suite_display}[/]")

        # Columns: metric | claude-stand | claude+memo | Δ | gpt-stand | gpt+memo | Δ
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        t.add_column("Metric",              style="dim",    min_width=20)
        t.add_column("Claude\nStandalone",  justify="center")
        t.add_column("Claude\n+AgenticMemo",justify="center")
        t.add_column("Δ Claude",            justify="center", min_width=9)
        t.add_column("GPT\nStandalone",     justify="center")
        t.add_column("GPT\n+AgenticMemo",   justify="center")
        t.add_column("Δ GPT",               justify="center", min_width=9)

        def _get(config_label: str) -> SuiteResult | None:
            return results.get(config_label, {}).get(suite_name)

        cs = _get("Claude Standalone")
        cm = _get("Claude + AgenticMemo")
        gs = _get("GPT Standalone")
        gm = _get("GPT + AgenticMemo")

        def _fmt_sr(r: SuiteResult | None) -> str:
            if r is None or r.tasks_total == 0: return "—"
            c = _color(r.success_rate)
            return f"[{c}]{r.success_rate:.1%}[/]"

        def _fmt_val(r: SuiteResult | None, attr: str, fmt: str = ".1f") -> str:
            if r is None: return "—"
            v = getattr(r, attr, 0)
            if v == 0 and attr not in ("avg_reward",): return "—"
            return f"{v:{fmt}}"

        t.add_row(
            "Success Rate",
            _fmt_sr(cs), _fmt_sr(cm),
            _delta_str(cm.success_rate, cs.success_rate) if cs and cm else "—",
            _fmt_sr(gs), _fmt_sr(gm),
            _delta_str(gm.success_rate, gs.success_rate) if gs and gm else "—",
        )
        t.add_row(
            "Avg Steps",
            _fmt_val(cs, "avg_steps"), _fmt_val(cm, "avg_steps"),
            _delta_str(cm.avg_steps if cm else 0, cs.avg_steps if cs else 0, pct=False) if cs and cm else "—",
            _fmt_val(gs, "avg_steps"), _fmt_val(gm, "avg_steps"),
            _delta_str(gm.avg_steps if gm else 0, gs.avg_steps if gs else 0, pct=False) if gs and gm else "—",
        )
        t.add_row(
            "Avg Tokens",
            _fmt_val(cs, "avg_tokens", ".0f"), _fmt_val(cm, "avg_tokens", ".0f"),
            _delta_str(cm.avg_tokens if cm else 0, cs.avg_tokens if cs else 0, pct=False) if cs and cm else "—",
            _fmt_val(gs, "avg_tokens", ".0f"), _fmt_val(gm, "avg_tokens", ".0f"),
            _delta_str(gm.avg_tokens if gm else 0, gs.avg_tokens if gs else 0, pct=False) if gs and gm else "—",
        )
        t.add_row(
            "Avg Duration",
            _fmt_val(cs, "avg_duration_ms", ".0f") + (" ms" if cs else ""),
            _fmt_val(cm, "avg_duration_ms", ".0f") + (" ms" if cm else ""),
            "—",
            _fmt_val(gs, "avg_duration_ms", ".0f") + (" ms" if gs else ""),
            _fmt_val(gm, "avg_duration_ms", ".0f") + (" ms" if gm else ""),
            "—",
        )
        t.add_row(
            "Avg Reward",
            _fmt_val(cs, "avg_reward", ".3f"), _fmt_val(cm, "avg_reward", ".3f"),
            _delta_str(cm.avg_reward if cm else 0, cs.avg_reward if cs else 0, pct=False) if cs and cm else "—",
            _fmt_val(gs, "avg_reward", ".3f"), _fmt_val(gm, "avg_reward", ".3f"),
            _delta_str(gm.avg_reward if gm else 0, gs.avg_reward if gs else 0, pct=False) if gs and gm else "—",
        )
        console.print(t)

        # Suite-specific extra metrics
        for r in [cm, gm]:
            if r and r.extra:
                extras = {k: v for k, v in r.extra.items()
                          if not isinstance(v, (list, dict)) and k != "note"}
                if extras:
                    console.print(f"  [dim][{r.config_label}] extras:[/] " +
                                  ", ".join(f"{k.replace('_',' ')}={v}"
                                            for k, v in list(extras.items())[:4]))

    # ── Overall summary ──────────────────────────────────────────────────────
    console.print("\n")
    console.rule("[bold green]Overall Summary")

    configs_ordered = [
        "Claude Standalone", "Claude + AgenticMemo",
        "GPT Standalone",    "GPT + AgenticMemo",
    ]

    t2 = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    t2.add_column("Configuration",  min_width=24)
    t2.add_column("Total Tasks",   justify="center")
    t2.add_column("Success Rate",  justify="center")
    t2.add_column("Avg Steps",     justify="center")
    t2.add_column("Avg Tokens",    justify="center")
    t2.add_column("Avg Reward",    justify="center")
    t2.add_column("vs Standalone", justify="center", min_width=12)

    standalone_rates = {
        "anthropic": None,
        "openai":    None,
    }

    rows = []
    for label in configs_ordered:
        all_r = list(results.get(label, {}).values())
        if not all_r:
            continue
        total   = sum(r.tasks_total for r in all_r if r.tasks_total)
        success = sum(r.tasks_success for r in all_r)
        sr      = success / total if total else 0.0
        steps   = sum(r.avg_steps * r.tasks_total for r in all_r if r.tasks_total)
        tokens  = sum(r.avg_tokens * r.tasks_total for r in all_r if r.tasks_total)
        reward  = sum(r.avg_reward * r.tasks_total for r in all_r if r.tasks_total)
        rows.append((label, total, sr,
                     steps / max(total, 1),
                     tokens / max(total, 1),
                     reward / max(total, 1)))

    # Record standalone baselines
    for label, total, sr, avg_steps, avg_tokens, avg_reward in rows:
        if "Standalone" in label:
            prov = "anthropic" if "Claude" in label else "openai"
            standalone_rates[prov] = sr

    for label, total, sr, avg_steps, avg_tokens, avg_reward in rows:
        ccolor = _CONFIG_COLORS.get(label, "white")
        color  = _color(sr)
        # Compute improvement vs standalone
        if "+AgenticMemo" in label:
            prov = "anthropic" if "Claude" in label else "openai" if "GPT" in label else None
            base_sr = standalone_rates.get(prov) if prov else None
            if base_sr is not None:
                diff = sr - base_sr
                dcolor = "green" if diff > 0 else "red" if diff < 0 else "dim"
                vs = f"[{dcolor}]{diff*100:+.1f}pp[/]"
            else:
                vs = "—"
        else:
            vs = "(baseline)"

        t2.add_row(
            f"[{ccolor}]{label}[/]",
            str(total),
            f"[{color}]{sr:.1%}[/]",
            f"{avg_steps:.1f}",
            f"{avg_tokens:.0f}",
            f"{avg_reward:.3f}",
            vs,
        )

    console.print(t2)
    console.print()


def print_summary(results: list[SuiteResult], label: str = "") -> None:
    console.rule(f"[bold cyan]Summary{' — ' + label if label else ''}")
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    t.add_column("Suite",        min_width=28)
    t.add_column("Tasks",        justify="center")
    t.add_column("Success Rate", justify="center")
    t.add_column("Avg Steps",    justify="center")
    t.add_column("Avg Duration", justify="center")
    t.add_column("Avg Reward",   justify="center")
    t.add_column("Status",       justify="center")
    for r in results:
        sr = r.success_rate
        c = _color(sr)
        status = "PASS" if sr >= 0.85 else "WARN" if sr >= 0.60 else "FAIL"
        t.add_row(
            r.name,
            str(r.tasks_total) if r.tasks_total else "—",
            f"[{c}]{sr:.1%}[/]" if r.tasks_total else "—",
            f"{r.avg_steps:.1f}" if r.tasks_total else "—",
            f"{r.avg_duration_ms:.0f}ms" if r.tasks_total else "—",
            f"{r.avg_reward:.3f}" if r.tasks_total else "—",
            f"[{c}]{status}[/]",
        )
    console.print(t)
    total   = sum(r.tasks_total for r in results if r.tasks_total)
    success = sum(r.tasks_success for r in results)
    sr = success / total if total else 0
    console.print(
        f"\n[bold]Overall:[/] {success}/{total} tasks succeeded "
        f"([{'green' if sr >= 0.8 else 'yellow'}]{sr:.1%}[/])\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Research mode runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_research(args: argparse.Namespace) -> None:
    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = args.openai_key    or os.environ.get("OPENAI_API_KEY")

    # Filter configs based on available keys (and --gpt-only flag)
    gpt_only = getattr(args, "gpt_only", False)
    active_configs: list[tuple[str, str, bool]] = []
    for label, provider, memo in RESEARCH_CONFIGS:
        if gpt_only and provider != "openai":
            continue  # skip all non-GPT configs
        if provider == "mock":
            active_configs.append((label, provider, memo))
        elif provider == "anthropic" and anthropic_key:
            active_configs.append((label, provider, memo))
        elif provider == "openai" and openai_key:
            active_configs.append((label, provider, memo))
        else:
            console.print(f"[yellow]Skipping '{label}' — no API key for {provider}[/]")

    _hard_suites = ["hard_core", "hard_transfer", "hard_reflexion"]
    if args.suite == "all":
        suites_to_run = list(RESEARCH_SUITE_MAP.items())
    elif args.suite == "hard":
        suites_to_run = [(s, RESEARCH_SUITE_MAP[s]) for s in _hard_suites]
    else:
        suites_to_run = [(args.suite, RESEARCH_SUITE_MAP[args.suite])]
    suite_names = [s for s, _ in suites_to_run]

    config_labels = [label for label, _, _ in active_configs]
    hard_note = " [bold yellow](HARD MODE — 18 algorithm/system-design problems)[/]" if "hard_core" in suite_names else ""
    console.print(Panel(
        f"[bold cyan]AgenticMemo — Research Benchmark[/]{hard_note}\n"
        f"[dim]Configurations : [white]{', '.join(config_labels)}[/]\n"
        f"Suites        : [white]{', '.join(suite_names)}[/]\n"
        f"Goal          : [white]Standalone vs +AgenticMemo improvement (real LLMs only)[/][/]",
        title="AgenticMemo Research",
        border_style="cyan",
    ))

    # results[config_label][suite_name] = SuiteResult
    all_results: dict[str, dict[str, SuiteResult]] = {label: {} for label, _, _ in active_configs}

    for suite_name, suite_fn in suites_to_run:
        console.rule(f"[bold yellow]Suite: {suite_name.upper()}")
        for label, provider, memo in active_configs:
            api_key = anthropic_key if provider == "anthropic" else openai_key if provider == "openai" else None
            ccolor = _CONFIG_COLORS.get(label, "white")
            console.print(f"  [{ccolor}]▶ {label}[/]")
            t0 = time.time()
            try:
                result = await suite_fn(label, provider, api_key, memo, args.model)
            except Exception as e:
                console.print(f"  [red]Failed: {e}[/]")
                import traceback; traceback.print_exc()
                result = SuiteResult(name=suite_name, config_label=label, extra={"error": str(e)})
            elapsed = time.time() - t0
            console.print(f"  [dim]Done in {elapsed:.1f}s[/]")
            print_suite_result(result)
            all_results[label][suite_name] = result

    print_research_comparison(all_results, suite_names)

    if args.output:
        report = {
            "mode": "research",
            "configurations": [
                {"label": l, "provider": p, "agenticmemo": m}
                for l, p, m in active_configs
            ],
            "suites": suite_names,
            "results": {
                label: {
                    sname: {
                        "name": r.name,
                        "config_label": r.config_label,
                        "tasks_total": r.tasks_total,
                        "tasks_success": r.tasks_success,
                        "success_rate": round(r.success_rate, 4),
                        "avg_steps": round(r.avg_steps, 2),
                        "avg_tokens": round(r.avg_tokens, 2),
                        "avg_duration_ms": round(r.avg_duration_ms, 2),
                        "avg_reward": round(r.avg_reward, 4),
                        "extra": {k: v for k, v in r.extra.items()
                                  if not isinstance(v, list)},
                    }
                    for sname, r in suite_results.items()
                }
                for label, suite_results in all_results.items()
            },
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        console.print(f"[green]Research report saved → {args.output}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Compare mode (quick, all providers with AgenticMemo)
# ─────────────────────────────────────────────────────────────────────────────

async def run_compare(args: argparse.Namespace) -> None:
    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = args.openai_key    or os.environ.get("OPENAI_API_KEY")

    providers: dict[str, str | None] = {"mock": None}
    if anthropic_key:
        providers["anthropic"] = anthropic_key
    else:
        console.print("[yellow]ANTHROPIC_API_KEY not set — skipping[/]")
    if openai_key:
        providers["openai"] = openai_key
    else:
        console.print("[yellow]OPENAI_API_KEY not set — skipping[/]")

    suites_to_run = (
        list(SUITE_MAP.items()) if args.suite == "all"
        else [(args.suite, SUITE_MAP[args.suite])]
    )

    provider_list = ", ".join(_PROVIDER_LABELS.get(p, p) for p in providers)
    console.print(Panel(
        "[bold cyan]AgenticMemo Benchmark — Provider Comparison[/]\n"
        f"[dim]Providers : [white]{provider_list}[/]\n"
        f"Suite(s)  : [white]{args.suite}[/][/]",
        title="AgenticMemo",
        border_style="cyan",
    ))

    all_results: dict[str, list[SuiteResult]] = {}
    for prov, key in providers.items():
        console.rule(f"[{_PROVIDER_COLORS.get(prov,'white')}]{_PROVIDER_LABELS.get(prov,prov)}[/]")
        results = []
        for sname, sfn in suites_to_run:
            console.rule(f"[bold yellow]  Suite: {sname.upper()}")
            t0 = time.time()
            try:
                r = await sfn(prov, key, args.model)
                r.config_label = prov
            except Exception as e:
                console.print(f"[red]Suite '{sname}' failed for {prov}: {e}[/]")
                import traceback; traceback.print_exc()
                r = SuiteResult(name=sname, config_label=prov, extra={"error": str(e)})
            console.print(f"[dim]Done in {time.time()-t0:.1f}s[/]")
            print_suite_result(r)
            results.append(r)
        print_summary(results, _PROVIDER_LABELS.get(prov, prov))
        all_results[prov] = results

    # Simple side-by-side
    console.rule("[bold green]Provider Comparison")
    for sname, _ in suites_to_run:
        t = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", title=sname.upper())
        t.add_column("Metric", style="dim", min_width=20)
        for prov in providers:
            pc = _PROVIDER_COLORS.get(prov, "white")
            t.add_column(f"[{pc}]{_PROVIDER_LABELS.get(prov,prov)}[/]", justify="center")
        for attr, label, fmt in [
            ("success_rate",   "Success Rate",  ".1%"),
            ("avg_steps",      "Avg Steps",     ".1f"),
            ("avg_tokens",     "Avg Tokens",    ".0f"),
            ("avg_duration_ms","Avg Duration",  ".0f"),
            ("avg_reward",     "Avg Reward",    ".3f"),
        ]:
            row = [label]
            for prov in providers:
                r = next((x for x in all_results[prov] if x.name and sname in x.name.lower()), None)
                if r is None:
                    row.append("—")
                    continue
                v = getattr(r, attr, 0)
                if attr == "success_rate":
                    row.append(f"[{_color(v)}]{v:{fmt}}[/]")
                else:
                    row.append(f"{v:{fmt}}")
            t.add_row(*row)
        console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# Single-provider mode
# ─────────────────────────────────────────────────────────────────────────────

async def run_single(args: argparse.Namespace) -> None:
    model_label = args.model or (
        "claude-sonnet-4-6" if args.provider == "anthropic"
        else "gpt-4o" if args.provider == "openai"
        else "mock-llm-v1"
    )
    console.print(Panel(
        "[bold cyan]AgenticMemo Benchmark[/]\n"
        f"[dim]Provider : [white]{args.provider}[/]\n"
        f"Model    : [white]{model_label}[/]\n"
        f"Suite(s) : [white]{args.suite}[/][/]",
        title="AgenticMemo",
        border_style="cyan",
    ))

    suites_to_run = (
        list(SUITE_MAP.items()) if args.suite == "all"
        else [(args.suite, SUITE_MAP[args.suite])]
    )
    results = []
    for sname, sfn in suites_to_run:
        console.rule(f"[bold yellow]Suite: {sname.upper()}")
        t0 = time.time()
        try:
            r = await sfn(args.provider, args.api_key, args.model)
            r.config_label = args.provider
        except Exception as e:
            console.print(f"[red]{sname} failed: {e}[/]")
            import traceback; traceback.print_exc()
            r = SuiteResult(name=sname, config_label=args.provider, extra={"error": str(e)})
        console.print(f"[dim]Done in {time.time()-t0:.1f}s[/]")
        print_suite_result(r)
        results.append(r)

    print_summary(results, args.provider)

    if args.output:
        report = {
            "mode": "single",
            "provider": args.provider,
            "suites": [
                {
                    "name": r.name,
                    "success_rate": round(r.success_rate, 4),
                    "avg_steps": round(r.avg_steps, 2),
                    "avg_duration_ms": round(r.avg_duration_ms, 2),
                    "avg_reward": round(r.avg_reward, 4),
                    "extra": {k: v for k, v in r.extra.items() if not isinstance(v, list)},
                }
                for r in results
            ],
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        console.print(f"[green]Report saved → {args.output}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    if args.research or getattr(args, "gpt_only", False):
        await run_research(args)
    elif args.compare:
        await run_compare(args)
    else:
        await run_single(args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgenticMemo Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Modes ────────────────────────────────────────────────────────────────
    p.add_argument("--research", action="store_true",
                   help="Research mode: Standalone vs +AgenticMemo for each LLM (for paper)")
    p.add_argument("--compare", action="store_true",
                   help="Compare all available providers (all with AgenticMemo)")
    p.add_argument("--gpt-only", action="store_true", dest="gpt_only",
                   help="Research mode: skip Claude, run only GPT Standalone vs GPT+AgenticMemo")

    # ── API keys ─────────────────────────────────────────────────────────────
    p.add_argument("--anthropic-key", default=None,
                   help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    p.add_argument("--openai-key", default=None,
                   help="OpenAI API key (overrides OPENAI_API_KEY env var)")

    # ── Single-provider ───────────────────────────────────────────────────────
    p.add_argument("--provider", default="mock",
                   choices=["mock", "anthropic", "openai"])
    p.add_argument("--api-key", default=None, dest="api_key")
    p.add_argument("--model", default=None,
                   help="Override model (e.g. claude-opus-4-6, gpt-4o-mini)")

    # ── Common ───────────────────────────────────────────────────────────────
    p.add_argument("--suite", default="all",
                   choices=["all", "hard"] + list(RESEARCH_SUITE_MAP),
                   help="Suite to run (default: all; 'hard' = hard_core+hard_transfer+hard_reflexion)")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="Save JSON report to FILE")

    args = p.parse_args()

    # Auto-detect keys
    if args.api_key is None:
        if args.provider == "anthropic":
            args.api_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
        elif args.provider == "openai":
            args.api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")

    # Auto-upgrade single mode from mock if a key is found
    if not args.research and not args.compare and args.provider == "mock":
        if args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY"):
            args.provider = "anthropic"
            args.api_key  = args.anthropic_key or os.environ["ANTHROPIC_API_KEY"]
            console.print("[dim]Auto-detected ANTHROPIC_API_KEY — use --research for full paper comparison[/]")
        elif args.openai_key or os.environ.get("OPENAI_API_KEY"):
            args.provider = "openai"
            args.api_key  = args.openai_key or os.environ["OPENAI_API_KEY"]
            console.print("[dim]Auto-detected OPENAI_API_KEY — use --research for full paper comparison[/]")

    return args


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
