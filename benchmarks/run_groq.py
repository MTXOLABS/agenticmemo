"""Token-budgeted benchmark for small models on the Groq API.

Compares Standalone vs +AgenticMemo on the existing benchmark tasks while
tracking EVERY token (planner, executor, verifier, reflection, filter) against
a hard daily budget — designed for Groq's free-tier 1M tokens/day limit.

Token-efficiency choices:
  - Light task suites only (solo coding + repeat/transfer tasks). The heavy
    finance/real-estate tasks burn 30-60k tokens each on small models.
  - max_steps=6, max_tokens=1024, no reflexion retries by default.
  - Standalone arm skips the LLM quality filter (it stores nothing anyway).
  - Hard budget stop with per-task checkpointing: if the budget runs out
    mid-suite, everything completed so far is already saved.

Usage:
    export GROQ_API_KEY=gsk_...
    .venv/bin/python -m benchmarks.run_groq                      # light suite
    .venv/bin/python -m benchmarks.run_groq --suite micro        # 5-task smoke
    .venv/bin/python -m benchmarks.run_groq --provider mock      # free dry-run
    .venv/bin/python -m benchmarks.run_groq --model llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agenticmemo import Agent, AgentConfig, LearningConfig, MemoryConfig
from agenticmemo.llm.base import LLMBackend
from agenticmemo.llm.openai_llm import OpenAILLM
from agenticmemo.tools import PythonReplTool
from agenticmemo.types import LLMResponse, TaskStatus

from .mock_llm import MockLLM
from .tasks import (
    ALL_SOLO_TASKS,
    FINANCE_REPEAT_TASKS,
    FINANCE_TASKS,
    HARD_ALGO_TASKS,
    HARD_DP_TASKS,
    HARD_GRAPH_TASKS,
    HARD_REPEAT_TASKS,
    HARD_SYSDESIGN_TASKS,
    REAL_ESTATE_REPEAT_TASKS,
    REAL_ESTATE_TASKS,
    REPEAT_TASKS,
    BenchTask,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

console = Console()


class BudgetExceededError(Exception):
    """Raised by TokenMeter when the hard token budget is exhausted."""


class TokenMeter(LLMBackend):
    """Wraps any LLMBackend and counts every token that flows through it.

    Unlike Trajectory.total_tokens (executor calls only), this sees planner,
    verifier, reflection, and filter calls too — the true budget consumption.
    Raises BudgetExceededError before starting a call once the budget is spent.
    """

    def __init__(self, inner: LLMBackend, budget: int, tpm: int = 7000) -> None:
        super().__init__(model=inner.model)
        self._inner = inner
        self.budget = budget
        self.tpm = tpm                      # self-throttle below the provider TPM cap
        self.used = 0
        self.calls = 0
        self._window: list[tuple[float, int]] = []  # (timestamp, tokens) last 60s

    def _estimate_request(self, messages, system) -> int:
        """Groq admits requests by REQUESTED tokens: prompt + reserved output
        (max_tokens). Pacing must mirror that, not actual usage after the fact."""
        chars = len(system or "") + sum(len(m.content or "") for m in messages)
        prompt_est = chars // 4 + 400  # +400 for tool schemas / structure
        return prompt_est + getattr(self._inner, "max_tokens", 1300)

    async def _pace(self, est_tokens: int) -> None:
        """Sleep until the estimated next request fits in the per-minute window."""
        while True:
            now = time.time()
            self._window = [(t, n) for t, n in self._window if now - t < 60]
            if not self._window or sum(n for _, n in self._window) + est_tokens <= self.tpm:
                return
            wait = 60 - (now - self._window[0][0]) + 0.5
            await asyncio.sleep(max(1.0, min(wait, 61.0)))

    @staticmethod
    def _err_text(e: BaseException) -> str:
        """Unwrap tenacity RetryError / exception chains to the root message."""
        last = getattr(e, "last_attempt", None)
        if last is not None and last.exception() is not None:
            e = last.exception()
        while e.__cause__ is not None:
            e = e.__cause__
        return str(e)

    async def complete(self, messages, tools=None, system=None) -> LLMResponse:
        if self.used >= self.budget:
            raise BudgetExceededError(f"Token budget exhausted: {self.used}/{self.budget}")
        est = self._estimate_request(messages, system)
        await self._pace(est)
        last_exc: BaseException | None = None
        for _ in range(25):  # 429s are a time cost, not a failure — be patient
            try:
                resp = await self._inner.complete(messages, tools=tools, system=system)
            except Exception as e:
                msg = self._err_text(e)
                if "429" not in msg and "rate limit" not in msg.lower():
                    raise
                # Groq tells us exactly how long to wait: "try again in 12.41s"
                m = re.search(r"try again in ([\d.]+)", msg)
                wait = min(90.0, float(m.group(1)) + 1.0) if m else 20.0
                console.print(f"[yellow]rate-limited, waiting {wait:.0f}s[/]")
                await asyncio.sleep(wait)
                last_exc = e
                continue
            self.used += resp.input_tokens + resp.output_tokens
            self.calls += 1
            # Record the REQUESTED size in the window — that's what Groq counts
            self._window.append((time.time(), est))
            return resp
        raise last_exc  # rate-limit retries exhausted

    async def embed(self, texts):
        return await self._inner.embed(texts)

    def format_tools(self, tools):
        # Delegate so provider-specific schema conversion is preserved
        if hasattr(self._inner, "format_tools"):
            return self._inner.format_tools(tools)
        return tools


def _make_llm(args: argparse.Namespace) -> LLMBackend:
    if args.provider == "mock":
        return MockLLM()
    env = "GROQ_API_KEY" if args.provider == "groq" else "OPENAI_API_KEY"
    api_key = args.api_key or os.environ.get(env)
    if not api_key:
        raise SystemExit(f"{env} not set (or pass --api-key)")
    return OpenAILLM(
        model=args.model,
        api_key=api_key,
        base_url=GROQ_BASE_URL if args.provider == "groq" else None,
        max_tokens=args.max_tokens,
        temperature=0.0,
    )


def _agent_config(standalone: bool, args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        max_steps=args.max_steps,
        max_retries=args.retries,
        memory=MemoryConfig(),  # in-memory only; standalone arm discards its agent
        learning=LearningConfig(
            enable_verification=True,          # honest success signal (never disable)
            max_reflexion_retries=args.retries,
            # Standalone stores nothing that outlives the task, so the extra
            # LLM self-eval call would be pure token waste.
            enable_quality_filter=not standalone,
        ),
    )


def _keyword_hits(task: BenchTask, answer: str) -> float:
    if not task.expected_keywords:
        return 1.0
    text = answer.lower()
    hits = sum(1 for kw in task.expected_keywords if kw.lower().replace("_", " ") in text
               or kw.lower() in text)
    return hits / len(task.expected_keywords)


# Deterministic numeric grading (Phase 5.1): mechanical, judge-free scoring.
# Accepts any common representation of a value — percent vs fraction,
# $M vs raw, thousands separators — within ±1.5% relative tolerance.
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")
_SCALES = (1.0, 1e-2, 1e2, 1e-3, 1e3, 1e-6, 1e6, 1e-9, 1e9)


def _numeric_hits(task: BenchTask, text: str) -> float | None:
    """Fraction of expected numbers present in output; None if task has none."""
    if not task.expected_numbers:
        return None
    found = []
    for m in _NUM_RE.finditer(text):
        try:
            found.append(float(m.group().replace(",", "")))
        except ValueError:
            continue
    hits = 0
    for e in task.expected_numbers:
        if any(
            abs(n - e * s) <= abs(e * s) * 0.015
            for n in found for s in _SCALES if e * s != 0
        ):
            hits += 1
    return hits / len(task.expected_numbers)


async def _run_arm(
    arm: str,
    tasks: list[BenchTask],
    meter: TokenMeter,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    out_path: Path,
    transfer_from: int = 10**9,
) -> None:
    """Run one arm. `standalone` = fresh agent per task; `memory` = one agent."""
    standalone = arm == "standalone"
    shared_agent: Agent | None = None
    if not standalone:
        shared_agent = Agent(meter, _agent_config(False, args))
        shared_agent.add_tool(PythonReplTool())
        if args.warm_memory:
            n = await shared_agent.load_memory_pack(args.warm_memory)
            console.print(f"[green]memory arm warmed with {n} curated cases[/]")

    results = checkpoint["arms"].setdefault(arm, [])

    for i, task in enumerate(tasks):
        if any(r["task_index"] == i for r in results):
            continue  # already done (resumed run)

        agent = shared_agent
        if standalone:
            agent = Agent(meter, _agent_config(True, args))
            agent.add_tool(PythonReplTool())

        tokens_before = meter.used
        t0 = time.time()
        err_msg = None
        breakdown: dict[str, int] = {}
        try:
            traj = await agent.run(task.task)
            status = traj.status.value
            # Accuracy text = final answer + tool observations, so keyword
            # evidence in printed code output counts, not just the summary.
            answer = traj.final_answer + " " + " ".join(
                s.observation for s in traj.steps if s.observation
            )
            steps = traj.num_steps
            breakdown = traj.metadata.get("token_breakdown", {})
        except BudgetExceededError:
            checkpoint["budget_exhausted_during"] = f"{arm}/task-{i}"
            _save(checkpoint, out_path)
            raise
        except Exception as e:  # single-task failure shouldn't kill the suite
            root = e
            while root.__cause__ is not None:
                root = root.__cause__
            err_msg = f"{type(root).__name__}: {str(root)[:300]}"
            status, answer, steps = "error", "", 0

        results.append({
            "task_index": i,
            "category": task.category,
            "transfer": i >= transfer_from,
            "task": task.task[:120],
            "status": status,
            "verified_success": status == TaskStatus.SUCCESS.value,
            "keyword_hit_rate": round(_keyword_hits(task, answer), 3),
            **(
                {"numeric_accuracy": round(na, 3)}
                if (na := _numeric_hits(task, answer)) is not None else {}
            ),
            "steps": steps,
            "tokens": meter.used - tokens_before,
            "seconds": round(time.time() - t0, 1),
            **({"breakdown": breakdown} if breakdown else {}),
            **({"error": err_msg} if err_msg else {}),
        })
        _save(checkpoint, out_path)  # checkpoint after every task

        done = sum(1 for r in results if r)
        console.print(
            f"[{'cyan' if standalone else 'green'}]{arm}[/] "
            f"{done}/{len(tasks)} | {task.category:8s} | {status:8s} "
            f"| {meter.used - tokens_before:>6d} tok | budget {meter.used}/{meter.budget}"
        )
        if args.sleep and args.provider != "mock":
            await asyncio.sleep(args.sleep)  # stay under Groq free-tier RPM/TPM


def _save(checkpoint: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, indent=2))
    tmp.replace(out_path)


def _summarize(checkpoint: dict[str, Any]) -> None:
    table = Table(title=f"Groq benchmark — {checkpoint['model']}")
    for col in ("Arm", "Tasks", "Verified success", "Keyword hit", "Avg steps", "Tokens"):
        table.add_column(col)

    summary: dict[str, Any] = {}
    for arm, rows in checkpoint["arms"].items():
        if not rows:
            continue
        n = len(rows)
        succ = sum(r["verified_success"] for r in rows)
        kw = sum(r["keyword_hit_rate"] for r in rows) / n
        steps = sum(r["steps"] for r in rows) / n
        toks = sum(r["tokens"] for r in rows)
        summary[arm] = {
            "tasks": n, "success": succ, "success_rate": round(succ / n, 3),
            "keyword_hit_rate": round(kw, 3), "avg_steps": round(steps, 2),
            "tokens": toks,
        }
        table.add_row(arm, f"{n}", f"{succ}/{n} ({succ / n:.0%})",
                      f"{kw:.0%}", f"{steps:.1f}", f"{toks:,}")

    checkpoint["summary"] = summary
    console.print(table)

    # Accuracy breakdown: per category, and transfer-only (the memory signal)
    def _acc(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "—"
        v = sum(r["verified_success"] for r in rows) / len(rows)
        k = sum(r["keyword_hit_rate"] for r in rows) / len(rows)
        graded = [r["numeric_accuracy"] for r in rows if "numeric_accuracy" in r]
        num = f" | numeric {sum(graded)/len(graded):.0%}" if graded else ""
        return f"verified {v:.0%} | keyword {k:.0%}{num} (n={len(rows)})"

    breakdown: dict[str, Any] = {}
    for arm, rows in checkpoint["arms"].items():
        if not rows:
            continue
        cats = sorted({r["category"] for r in rows})
        breakdown[arm] = {
            **{c: _acc([r for r in rows if r["category"] == c]) for c in cats},
            "transfer_only": _acc([r for r in rows if r.get("transfer")]),
        }
        console.print(f"\n[bold]{arm} accuracy[/]")
        for label, line in breakdown[arm].items():
            console.print(f"  {label:14s} {line}")
    checkpoint["accuracy_breakdown"] = breakdown

    if "standalone" in summary and "memory" in summary:
        delta = summary["memory"]["success_rate"] - summary["standalone"]["success_rate"]
        console.print(f"\n[bold]Δ success (memory − standalone): {delta:+.0%}[/]")


async def main() -> None:
    p = argparse.ArgumentParser(description="Token-budgeted Groq benchmark")
    p.add_argument("--provider", choices=["groq", "openai", "mock"], default="groq")
    p.add_argument("--model", default=None,
                   help="default: openai/gpt-oss-120b (groq) / gpt-5-mini (openai)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--budget", type=int, default=600_000,
                   help="hard token cap for this run (default 600k of the 1M/day)")
    p.add_argument("--tpm", type=int, default=None,
                   help="tokens-per-minute self-throttle "
                        "(default: 7000 on groq free tier, 150000 on openai)")
    p.add_argument("--suite",
                   choices=["light", "micro", "domain", "domain-micro", "hard"],
                   default="light",
                   help="light = solo + repeat; micro = 3+2 smoke; "
                        "domain = 4 finance + 4 real-estate + 3 transfer; "
                        "domain-micro = 2+2+1 smoke; "
                        "hard = 3 each of dp/graph/sysdesign/algo + 4 transfer")
    p.add_argument("--max-steps", type=int, default=None,
                   help="default: 6 (light suites) / 8 (domain suites)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="default: 1024 (light) / 1300 (domain — long code outputs)")
    p.add_argument("--retries", type=int, default=0,
                   help="reflexion retries per task (each retry ~doubles task cost)")
    p.add_argument("--sleep", type=float, default=2.0,
                   help="seconds between tasks (free-tier rate limits)")
    p.add_argument("--output", default=None)
    p.add_argument("--warm-memory", default=None, metavar="SEEDS_JSON",
                   help="pre-load the memory arm with curated cases "
                        "(e.g. benchmarks/seeds/domain_micro.json)")
    args = p.parse_args()

    # Per-provider defaults
    if args.model is None:
        args.model = "openai/gpt-oss-120b" if args.provider == "groq" else "gpt-5-mini"
    if args.tpm is None:
        args.tpm = 7000 if args.provider == "groq" else 150_000

    # Suite selection. Transfer tasks come LAST so the memory arm has seed
    # experience to draw on — that ordering is what makes the accuracy delta
    # on transfer tasks the key memory signal.
    if args.suite == "micro":
        seed, transfer = ALL_SOLO_TASKS[:3], REPEAT_TASKS[:2]
    elif args.suite == "light":
        seed, transfer = ALL_SOLO_TASKS, REPEAT_TASKS
    elif args.suite == "domain-micro":
        # 4 questions/arm: enough for a real signal without burning the budget
        seed = FINANCE_TASKS[:2] + REAL_ESTATE_TASKS[:1]
        transfer = FINANCE_REPEAT_TASKS[:1]
    elif args.suite == "hard":
        seed = (HARD_DP_TASKS[:3] + HARD_GRAPH_TASKS[:3]
                + HARD_SYSDESIGN_TASKS[:3] + HARD_ALGO_TASKS[:3])
        transfer = HARD_REPEAT_TASKS[:4]
    else:  # domain
        seed = FINANCE_TASKS[:4] + REAL_ESTATE_TASKS[:4]
        transfer = FINANCE_REPEAT_TASKS[:2] + REAL_ESTATE_REPEAT_TASKS[:1]
    tasks = seed + transfer
    transfer_from = len(seed)

    # Domain/hard tasks are multi-step computations — they need more steps and
    # longer code outputs than the light coding tasks.
    is_domain = args.suite.startswith("domain") or args.suite == "hard"
    if args.max_steps is None:
        args.max_steps = 8 if is_domain else 6
    if args.max_tokens is None:
        args.max_tokens = 1300 if is_domain else 1024

    out_path = Path(
        args.output
        or f"benchmarks/results/{args.provider}_{args.model.replace('/', '_')}"
           f"_{date.today():%Y%m%d}.json"
    )

    # Resume from checkpoint if one exists (budget may have run out yesterday)
    if out_path.exists():
        checkpoint = json.loads(out_path.read_text())
        # Errored tasks (rate-limit starvation etc.) are retryable — drop them
        # so resume re-runs them instead of freezing the error into results.
        for arm, rows in checkpoint.get("arms", {}).items():
            kept = [r for r in rows if r.get("status") != "error"]
            if len(kept) != len(rows):
                console.print(f"[yellow]{arm}: retrying {len(rows) - len(kept)} errored task(s)[/]")
            checkpoint["arms"][arm] = kept
        console.print(f"[yellow]Resuming from checkpoint {out_path}[/]")
    else:
        checkpoint = {
            "model": "mock" if args.provider == "mock" else args.model,
            "provider": args.provider,
            "suite": args.suite,
            "date": f"{date.today()}",
            "config": {"max_steps": args.max_steps, "max_tokens": args.max_tokens,
                       "retries": args.retries, "verification": True},
            "arms": {},
        }

    meter = TokenMeter(_make_llm(args), budget=args.budget, tpm=args.tpm)
    console.print(
        f"Model: [bold]{checkpoint['model']}[/] | suite: {args.suite} "
        f"({len(tasks)} tasks/arm) | budget: {args.budget:,} tokens | out: {out_path}"
    )

    try:
        await _run_arm("standalone", tasks, meter, args, checkpoint, out_path, transfer_from)
        await _run_arm("memory", tasks, meter, args, checkpoint, out_path, transfer_from)
    except BudgetExceededError as e:
        console.print(f"[red]STOPPED: {e} — partial results saved to {out_path}[/]")

    checkpoint["total_tokens_used"] = meter.used
    checkpoint["total_llm_calls"] = meter.calls
    _summarize(checkpoint)
    _save(checkpoint, out_path)
    console.print(f"Saved → {out_path} | total tokens: {meter.used:,} ({meter.calls} calls)")


if __name__ == "__main__":
    asyncio.run(main())
