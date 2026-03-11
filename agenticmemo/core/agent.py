"""AgenticMemo — the main Agent class.

Orchestrates the full Planner-Executor-Memory loop with all 4 phases
+ 3 AgenticMemo v2 innovations:

  ┌───────────────────────────────────────────────────────────────────────┐
  │                        AgenticMemo v2 Loop                            │
  │                                                                        │
  │  Task ──→ HintLibrary + SkillLibrary + FailurePatterns                │
  │                  + Retriever ──────────────→ Planner ──→ Executor    │
  │                       ↑                                │   ↑          │
  │                       │                        DMER: mid-exec         │
  │                       │                        memory refresh         │
  │                       │                                │              │
  │        HierarchicalMemory ←── QualityFilter ←─────────┘              │
  │                ↑                    ↑                                  │
  │           GRPOPolicy        ReflexionEngine (on fail)                 │
  │                ↑                                                       │
  │    HintExtractor + SkillConsolidator + FailureMiner (periodic)        │
  └───────────────────────────────────────────────────────────────────────┘

Learning mechanisms (no LLM weights ever updated):
  Phase 1 — Ensemble retrieval (semantic + BM25 + graph + temporal)
  Phase 2 — Reflexion failure loop + GRPO retrieval policy
  Phase 3 — Temporal Knowledge Graph + Hierarchical memory (H-MEM)
  Phase 4 — Multi-agent SharedMemoryPool + HintExtractor internalization

AgenticMemo v2 Innovations:
  CFM  — Causal Failure Mining: anti-case bank injected at planning time
  ESMC — Episodic-to-Semantic Memory Consolidation: skill distillation
  DMER — Dynamic Mid-Execution Retrieval: memory refresh at each N steps
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ..config import AgentConfig
from ..learning.failure_miner import FailureMiner, FailurePatternBank
from ..learning.filters import TrajectoryFilter
from ..learning.grpo import GRPOPolicy
from ..learning.hints import HintExtractor, HintLibrary
from ..learning.reflexion import ReflexionEngine
from ..learning.skill_consolidator import SkillConsolidator, SkillLibrary
from ..llm.base import LLMBackend
from ..llm.anthropic_llm import AnthropicLLM
from ..llm.openai_llm import OpenAILLM
from ..memory.case import Case, CaseOutcome
from ..memory.hierarchical import HierarchicalMemory
from ..retrieval.ensemble import EnsembleRetriever
from ..tools.registry import ToolRegistry
from ..tools.base import Tool
from ..types import MemoryDomain, TaskStatus, Trajectory
from .executor import Executor
from .planner import Planner


_console = Console()

# Periodic learning intervals (in stored cases)
_HINT_EXTRACT_EVERY = 20
_SKILL_CONSOLIDATE_EVERY = 10   # successes
_FAILURE_MINE_EVERY = 5         # failures


class Agent:
    """Advanced Memento-inspired agent — all 4 phases implemented.

    Phases:
      1. Ensemble retrieval  — semantic + BM25 + graph + temporal
      2. GRPO policy + Reflexion — better case selection + failure learning
      3. Hierarchical + graph memory — H-MEM 4-layer + temporal knowledge graph
      4. Multi-agent memory + hints internalization — SharedMemoryPool + HintExtractor

    Quick start::

        from agenticmemo import Agent
        from agenticmemo.tools import PythonReplTool

        agent = Agent.from_anthropic(api_key="sk-ant-...")
        agent.add_tool(PythonReplTool())

        result = await agent.run("Calculate the first 10 Fibonacci numbers")
        print(result.final_answer)
    """

    def __init__(
        self,
        llm: LLMBackend,
        cfg: AgentConfig | None = None,
    ) -> None:
        self._llm = llm
        self._cfg = cfg or AgentConfig()

        # Memory
        self._memory = HierarchicalMemory(self._cfg.memory)

        # Retrieval
        self._retriever = EnsembleRetriever(self._memory, self._cfg.retrieval)

        # Tools
        self._tools = ToolRegistry()

        # Learning — Phases 1-3
        self._filter = TrajectoryFilter(llm, self._cfg.learning)
        self._reflexion = ReflexionEngine(llm, self._cfg.learning)
        self._grpo = GRPOPolicy(self._cfg.retrieval)

        # Learning — Phase 4: hints internalization
        self._hint_library = HintLibrary()
        self._hint_extractor = HintExtractor(llm, self._hint_library)
        self._cases_since_hint_extract: int = 0

        # v2: Causal Failure Mining (CFM) — persists alongside main memory
        _base = self._cfg.memory.persist_path
        _failure_path = (_base.replace(".json", "_failures.json") if _base else None)
        _skills_path  = (_base.replace(".json", "_skills.json")   if _base else None)

        self._failure_bank = FailurePatternBank(persist_path=_failure_path)
        self._failure_miner = FailureMiner(llm, self._failure_bank)

        # v2: Episodic-to-Semantic Memory Consolidation (ESMC)
        self._skill_library = SkillLibrary(persist_path=_skills_path)
        self._skill_consolidator = SkillConsolidator(llm, self._skill_library)

        # Core components
        # v2: pass retriever to executor for DMER (Dynamic Mid-Execution Retrieval)
        self._planner = Planner(llm, self._retriever, self._cfg)
        self._executor = Executor(
            llm, self._tools, self._cfg,
            retriever=self._retriever,       # DMER enabled
            memory_refresh_every=3,
            memory_refresh_top_k=2,
        )

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_anthropic(
        cls,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        cfg: AgentConfig | None = None,
        **llm_kwargs: Any,
    ) -> "Agent":
        llm = AnthropicLLM(model=model, api_key=api_key, **llm_kwargs)
        return cls(llm, cfg)

    @classmethod
    def from_openai(
        cls,
        api_key: str | None = None,
        model: str = "gpt-4o",
        cfg: AgentConfig | None = None,
        **llm_kwargs: Any,
    ) -> "Agent":
        llm = OpenAILLM(model=model, api_key=api_key, **llm_kwargs)
        return cls(llm, cfg)

    # ------------------------------------------------------------------ #
    # Tool management
    # ------------------------------------------------------------------ #

    def add_tool(self, tool: Tool) -> "Agent":
        self._tools.register(tool)
        return self

    def add_tools(self, *tools: Tool) -> "Agent":
        self._tools.register_many(*tools)
        return self

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def run(self, task: str, domain: MemoryDomain | None = None) -> Trajectory:
        """Run a task and return the final Trajectory.

        Full pipeline (all 4 phases):
          1. Retrieve relevant cases (ensemble)  +  inject hints (Phase 4)
          2. Plan (case-augmented + hints-augmented)
          3. Execute (ReAct loop)
          4. Reflexion (on failure, retry up to N times)   [Phase 2]
          5. Quality filter → store Case                   [Phase 1]
          6. GRPO update                                   [Phase 2]
          7. Periodic hint extraction                      [Phase 4]

        Args:
            task:   Natural-language task description.
            domain: Optional domain hint for targeted retrieval.
        """
        if self._cfg.verbose:
            _console.print(Panel(f"[bold cyan]Task:[/] {task}", title="AgenticMemo"))

        reflection: str | None = None
        trajectory: Trajectory | None = None

        for attempt in range(self._cfg.max_retries + 1):
            if attempt > 0 and self._cfg.verbose:
                _console.print(f"[yellow]Retry {attempt}/{self._cfg.max_retries}[/]")

            # Phase 4: build hints context for planner
            domain_str = domain.value if domain else None
            hints_block = self._hint_library.to_prompt_block(domain=domain_str)

            # v2 CFM: inject failure patterns as negative examples
            failure_patterns_block = self._failure_bank.to_prompt_block(
                domain=domain_str, top_n=4
            )

            # v2 ESMC: inject consolidated skills
            skills_block = self._skill_library.to_prompt_block(
                domain=domain_str, top_n=3
            )

            # 1+2. Plan (with cases + hints + failure patterns + skills)
            plan, retrieved_cases = await self._planner.plan(
                task,
                tool_schemas=self._tools.schemas(),
                reflection=reflection,
                hints=hints_block or None,
                failure_patterns=failure_patterns_block or None,
                skills=skills_block or None,
            )
            if self._cfg.verbose:
                _console.print(f"[dim]Plan:\n{plan[:400]}...[/]")

            # 3. Execute
            sys_prefix = (
                self._reflexion.build_retry_system(reflection) if reflection else None
            )
            trajectory = await self._executor.execute(task, plan, system_prefix=sys_prefix)

            if self._cfg.verbose:
                status_color = "green" if trajectory.status == TaskStatus.SUCCESS else "red"
                _console.print(
                    f"[{status_color}]Status: {trajectory.status.value}[/] "
                    f"| Steps: {trajectory.num_steps} "
                    f"| Tokens: {trajectory.total_tokens}"
                )

            # 4. Reflexion retry check
            if not self._reflexion.should_retry(trajectory, attempt):
                break

            reflection = await self._reflexion.reflect(task, trajectory)
            trajectory.reflection = reflection
            if self._cfg.verbose:
                _console.print(f"[yellow]Reflection: {reflection[:200]}[/]")

        assert trajectory is not None

        # 5. Assign reward
        reward = self._filter.assign_reward(trajectory)
        trajectory.reward = reward

        # Quality filter + store
        passed = await self._filter.filter_single(task, trajectory)
        if passed and reward >= self._cfg.memory.min_reward_to_store:
            await self._store_case(task, trajectory, retrieved_cases, domain)

        return trajectory

    # ------------------------------------------------------------------ #
    # Internal storage + learning
    # ------------------------------------------------------------------ #

    async def _store_case(
        self,
        task: str,
        trajectory: Trajectory,
        retrieved_cases: list[Case],
        domain: MemoryDomain | None,
    ) -> None:
        outcome = CaseOutcome(
            status=trajectory.status,
            reward=trajectory.reward,
            answer=trajectory.final_answer[:500],
            reflection=trajectory.reflection,
        )
        case = Case(
            task=task,
            domain=domain or MemoryDomain.GENERAL,
            trajectory=trajectory,
            outcome=outcome,
        )

        await self._memory.store(case)
        await self._retriever.index_case(case)

        # Graph edges to retrieved cases
        for prior in retrieved_cases:
            self._memory.graph.add_edge(case.id, prior.id)

        # GRPO update (Phase 2)
        self._grpo.record_outcome(
            case_ids=[c.id for c in retrieved_cases],
            reward=trajectory.reward,
        )
        all_cases = await self._memory.all_cases()
        case_map = {c.id: c for c in all_cases}
        updated = self._grpo.update(case_map)
        if updated and self._cfg.verbose:
            _console.print(f"[dim]GRPO updated {updated} case Q-values[/]")

        # Phase 4: periodic hint extraction
        self._cases_since_hint_extract += 1
        if self._cases_since_hint_extract >= _HINT_EXTRACT_EVERY:
            self._cases_since_hint_extract = 0
            domain_str = case.domain.value
            new_hints = await self._hint_extractor.extract(all_cases, domain=domain_str)
            if new_hints and self._cfg.verbose:
                _console.print(f"[dim]Extracted {len(new_hints)} new hints[/]")

        # v2 CFM: periodic failure pattern mining
        if trajectory.status == TaskStatus.SUCCESS:
            self._skill_consolidator.record_success()
        else:
            self._failure_miner.record_failure()

        if self._failure_miner.should_mine(_FAILURE_MINE_EVERY):
            new_patterns = await self._failure_miner.mine(all_cases)
            if new_patterns and self._cfg.verbose:
                _console.print(f"[dim]CFM: mined {len(new_patterns)} new failure patterns[/]")

        # v2 ESMC: periodic skill consolidation
        if self._skill_consolidator.should_consolidate(_SKILL_CONSOLIDATE_EVERY):
            new_skills = await self._skill_consolidator.consolidate(all_cases)
            if new_skills and self._cfg.verbose:
                _console.print(f"[dim]ESMC: distilled {len(new_skills)} new skills[/]")

    # ------------------------------------------------------------------ #
    # Memory management
    # ------------------------------------------------------------------ #

    async def memory_size(self) -> int:
        return await self._memory.size()

    async def clear_memory(self) -> None:
        await self._memory.clear()
        await self._retriever.rebuild_index()

    async def extract_hints_now(self, domain: str = "general") -> int:
        """Manually trigger hint extraction. Returns number of new hints."""
        cases = await self._memory.all_cases()
        new = await self._hint_extractor.extract(cases, domain=domain)
        return len(new)

    # ------------------------------------------------------------------ #
    # Stats & inspection
    # ------------------------------------------------------------------ #

    def grpo_stats(self) -> dict[str, float]:
        return self._grpo.stats()

    def hint_library(self) -> HintLibrary:
        return self._hint_library

    def skill_library(self) -> SkillLibrary:
        """v2 ESMC: access the consolidated skill library."""
        return self._skill_library

    def failure_bank(self) -> FailurePatternBank:
        """v2 CFM: access the failure pattern bank."""
        return self._failure_bank

    async def v2_stats(self) -> dict[str, int]:
        """Return counts of all v2 learning components."""
        return {
            "memory_cases": await self._memory.size(),
            "hints": self._hint_library.size(),
            "skills": self._skill_library.size(),          # ESMC
            "failure_patterns": self._failure_bank.size(), # CFM
        }

    async def recent_cases(self, n: int = 5) -> list[Case]:
        cases = await self._memory.all_cases()
        cases.sort(key=lambda c: c.created_at, reverse=True)
        return cases[:n]

    def __repr__(self) -> str:
        return (
            f"Agent(llm={self._llm!r}, tools={len(self._tools)}, "
            f"hints={self._hint_library.size()}, "
            f"skills={self._skill_library.size()}, "
            f"failure_patterns={self._failure_bank.size()})"
        )
