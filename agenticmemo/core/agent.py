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
from ..learning.verifier import OutcomeVerifier
from ..llm.accounting import TaggedLLM, TokenLedger
from ..llm.anthropic_llm import AnthropicLLM
from ..llm.base import LLMBackend
from ..llm.compat import apply_tool_quirks, warn_if_output_budget_low
from ..llm.openai_llm import OpenAILLM
from ..memory.case import Case, CaseOutcome, extract_solution
from ..memory.hierarchical import HierarchicalMemory
from ..retrieval.ensemble import EnsembleRetriever
from ..tools.base import Tool
from ..tools.registry import ToolRegistry
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
        judge_llm: LLMBackend | None = None,
    ) -> None:
        """Args:
            llm:       Primary backend for planning and execution.
            cfg:       Agent configuration.
            judge_llm: Optional cheaper backend for verification/filtering —
                       judging an answer is much easier than producing it, so
                       a nano-class judge cuts overhead without losing signal.
        """
        self._llm = llm
        self._cfg = cfg or AgentConfig()
        warn_if_output_budget_low(llm)

        # Per-subsystem token accounting: every component gets a tagged LLM
        # handle so each run's cost decomposes by purpose (Phase 0).
        self._ledger = TokenLedger()

        def tag(purpose: str, backend: LLMBackend = llm) -> TaggedLLM:
            return TaggedLLM(backend, purpose, self._ledger)

        judge = judge_llm or llm

        # Memory
        self._memory = HierarchicalMemory(self._cfg.memory)

        # Learning — Phases 1-3
        self._filter = TrajectoryFilter(tag("filter", judge), self._cfg.learning)
        self._verifier = OutcomeVerifier(tag("verifier", judge), self._cfg.learning)
        self._reflexion = ReflexionEngine(tag("reflexion"), self._cfg.learning)
        self._grpo = GRPOPolicy(self._cfg.retrieval)

        # Retrieval — GRPO policy is passed in so its learned Q-values
        # actually rerank retrieval results (planning + DMER).
        self._retriever = EnsembleRetriever(
            self._memory, self._cfg.retrieval, grpo_policy=self._grpo
        )

        # Tools
        self._tools = ToolRegistry()

        # Persistence paths for the auxiliary learning stores
        _base = self._cfg.memory.persist_path
        _failure_path = (_base.replace(".json", "_failures.json") if _base else None)
        _skills_path  = (_base.replace(".json", "_skills.json")   if _base else None)
        _hints_path   = (_base.replace(".json", "_hints.json")    if _base else None)

        # Learning — Phase 4: hints internalization
        self._hint_library = HintLibrary(persist_path=_hints_path)
        self._hint_library.load()
        self._hint_extractor = HintExtractor(tag("hints"), self._hint_library)
        self._cases_since_hint_extract: int = 0

        # v2: Causal Failure Mining (CFM) — persists alongside main memory
        self._failure_bank = FailurePatternBank(persist_path=_failure_path)
        self._failure_miner = FailureMiner(tag("mining"), self._failure_bank)

        # v2: Episodic-to-Semantic Memory Consolidation (ESMC)
        self._skill_library = SkillLibrary(persist_path=_skills_path)
        self._skill_consolidator = SkillConsolidator(tag("consolidation"), self._skill_library)

        # Core components
        # v2: pass retriever to executor for DMER (Dynamic Mid-Execution Retrieval)
        self._planner = Planner(tag("planner"), self._retriever, self._cfg)
        self._executor = Executor(
            tag("executor"), self._tools, self._cfg,
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
        judge_model: str | None = None,
        **llm_kwargs: Any,
    ) -> Agent:
        llm = AnthropicLLM(model=model, api_key=api_key, **llm_kwargs)
        judge = (
            AnthropicLLM(model=judge_model, api_key=api_key) if judge_model else None
        )
        return cls(llm, cfg, judge_llm=judge)

    @classmethod
    def from_openai(
        cls,
        api_key: str | None = None,
        model: str = "gpt-4o",
        cfg: AgentConfig | None = None,
        judge_model: str | None = None,
        **llm_kwargs: Any,
    ) -> Agent:
        llm = OpenAILLM(model=model, api_key=api_key, **llm_kwargs)
        judge = OpenAILLM(model=judge_model, api_key=api_key) if judge_model else None
        return cls(llm, cfg, judge_llm=judge)

    # ------------------------------------------------------------------ #
    # Tool management
    # ------------------------------------------------------------------ #

    def add_tool(self, tool: Tool) -> Agent:
        self._tools.register(apply_tool_quirks(tool, self._llm.model))
        return self

    def add_tools(self, *tools: Tool) -> Agent:
        for t in tools:
            self.add_tool(t)
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

        ledger_before = self._ledger.snapshot()
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
                domain=domain,
            )
            if self._cfg.verbose:
                _console.print(f"[dim]Plan:\n{plan[:400]}...[/]")

            # 3. Execute — with the best proven solution as an execution-time
            # exemplar. The planner summarizes memory, but a weaker model
            # benefits most from seeing working code WHILE executing: adapting
            # a proven artifact takes far fewer steps than re-deriving it.
            prefix_parts: list[str] = []
            exemplar = next(
                (c for c in retrieved_cases if c.is_success and c.solution), None
            )
            if exemplar is not None:
                # Template solutions (with an INPUTS block) turn adaptation
                # into a trivial edit — the easiest possible reuse for a
                # small model. Blob solutions fall back to free adaptation.
                if "# --- INPUTS ---" in exemplar.solution:
                    how = (
                        "Edit ONLY the `# --- INPUTS ---` block to match the "
                        "CURRENT task's numbers, keep the logic unchanged, "
                        "then EXECUTE it with your tools."
                    )
                else:
                    how = (
                        "Adapt this code to the CURRENT task's inputs and "
                        "EXECUTE it with your tools."
                    )
                prefix_parts.append(
                    "REFERENCE SOLUTION from a previously solved similar task "
                    f"(task: {exemplar.task[:150]}).\n{how} "
                    "Never copy its printed results as your answer — the "
                    "reference used different inputs; only your own execution "
                    "output counts:\n"
                    f"{exemplar.solution[:2000]}"
                )
            if reflection:
                prefix_parts.append(self._reflexion.build_retry_system(reflection))
            sys_prefix = "\n\n".join(prefix_parts) or None

            # Phase 3.2 — stepwise guidance: one-line "next action" scaffold
            # per turn from the exemplar's successful trajectory. Only for
            # genuinely multi-step exemplars: a 1-step pack case would emit
            # "→ final answer" as the FIRST hint, teaching the model to stop
            # before reporting its deliverables (measured: 2-step partials).
            exemplar_steps: list[str] | None = None
            if (
                exemplar is not None
                and self._cfg.enable_stepwise_guidance
                and len(exemplar.trajectory.steps) >= 3
            ):
                exemplar_steps = [
                    f"{s.thought[:100]}"
                    + (f" → {s.tool_call.name}" if s.tool_call else " → final answer")
                    for s in exemplar.trajectory.steps[:12]
                ]

            trajectory = await self._executor.execute(
                task, plan, system_prefix=sys_prefix, exemplar_steps=exemplar_steps
            )

            # 3.5 Verify outcome — the executor's SUCCESS only means "produced
            # a final answer"; the verifier judges whether it solves the task.
            trajectory.status = await self._verifier.verify(task, trajectory)

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

        # Quality filter + store. Successes must pass the full filter;
        # failures/partials only need to be structurally sane — they are
        # stored as anti-cases so the FailureMiner (CFM) can learn from them.
        # A verifier SUCCESS verdict subsumes the filter's LLM self-eval
        # (Phase 1.1 — one judgment per trajectory, not two).
        if trajectory.status == TaskStatus.SUCCESS:
            passed = await self._filter.filter_single(
                task, trajectory,
                verified=self._cfg.learning.enable_verification,
            )
        else:
            passed = self._filter.structural_ok(trajectory)
        if passed and reward >= self._cfg.memory.min_reward_to_store:
            # Lazy learning: ≤2-step runs carry no strategy worth mining —
            # store the case but skip the periodic learning machinery.
            await self._store_case(
                task, trajectory, retrieved_cases, domain,
                learn=trajectory.num_steps > 2,
                plan=plan,
            )

        trajectory.metadata["token_breakdown"] = self._ledger.delta_since(ledger_before)
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
        learn: bool = True,
        plan: str = "",
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
            solution=extract_solution(trajectory),
            # Plans are only worth reusing when they actually worked
            plan=plan if trajectory.status == TaskStatus.SUCCESS else "",
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

        if not learn:
            return  # trivial run: case stored + GRPO updated, no LLM learning

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

    async def load_memory_pack(self, path: str) -> int:
        """Pre-load expert-curated cases from a memory-pack JSON file.

        A memory pack is a list of entries::

            [{"task": "...", "domain": "finance",         # optional
              "code": "<verified working code>",
              "answer": "<its output>"}, ...]              # answer optional

        Packs let a strong model or human expert "teach" weaker models:
        solutions are stored as proven exemplars that retrieval surfaces on
        similar tasks. Reference outputs are withheld from prompts by design —
        models near their capability edge copy visible answers instead of
        executing (measured on three model families).

        Returns the number of cases loaded.
        """
        import json as _json  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415

        from ..types import Step, ToolCall, ToolResult  # noqa: PLC0415

        entries = _json.loads(_Path(path).read_text())
        for i, e in enumerate(entries):
            answer = e.get("answer", "")
            traj = Trajectory(
                task=e["task"], status=TaskStatus.SUCCESS, final_answer=answer
            )
            traj.add_step(Step(
                index=0,
                thought="Apply the verified reference implementation.",
                tool_call=ToolCall(id=f"pack{i}", name="python_repl",
                                   arguments={"code": e["code"]}),
                tool_result=ToolResult(tool_call_id=f"pack{i}",
                                       tool_name="python_repl", output=answer),
                observation="[reference executed successfully — output withheld]",
            ))
            case = Case(
                task=e["task"],
                domain=MemoryDomain(e["domain"]) if e.get("domain") else MemoryDomain.GENERAL,
                trajectory=traj,
                outcome=CaseOutcome(status=TaskStatus.SUCCESS, reward=1.0,
                                    answer=answer[:500]),
                solution=f"# via python_repl\n{e['code']}",
                plan=(
                    "Step 1: Take the proven reference implementation and edit "
                    "only its INPUTS to match this task.\n"
                    "Step 2: Execute the adapted code with the python tool.\n"
                    "Step 3: Check the task statement again — it lists MULTIPLE "
                    "required deliverables. If any are missing from your output, "
                    "extend the code and run again.\n"
                    "Step 4: Report EVERY requested value from your own "
                    "execution output. An answer missing deliverables is "
                    "incomplete."
                ),
            )
            await self._memory.store(case)
            await self._retriever.index_case(case)
        return len(entries)

    async def export_memory_pack(self, path: str) -> int:
        """Export solution-bearing successful cases as a memory-pack JSON.

        The exported pack can be loaded by any other agent — including ones
        running much smaller models. Returns the number of cases exported.
        """
        import json as _json  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415

        cases = await self._memory.all_cases()
        entries = []
        for c in cases:
            if not (c.is_success and c.solution):
                continue
            code = c.solution.split("\n", 1)[1] if "\n" in c.solution else c.solution
            entries.append({
                "task": c.task,
                "domain": c.domain.value,
                "code": code,
                "answer": c.outcome.answer,
            })
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(_json.dumps(entries, indent=2))
        tmp.replace(p)
        return len(entries)

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

    def token_breakdown(self) -> dict[str, dict[str, int]]:
        """Cumulative token/call counts per subsystem (planner, executor, ...)."""
        return self._ledger.breakdown()

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
