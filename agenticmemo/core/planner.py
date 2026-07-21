"""Case-augmented + hints-augmented Planner.

Stage 1 of the Planner-Executor loop:
  1. Retrieve top-K relevant cases from memory (ensemble retrieval).
  2. Inject internalized hints from the HintLibrary (Phase 4).
  3. Build a system prompt with cases + hints as "experience context".
  4. Ask the LLM for a structured step-by-step plan.
"""

from __future__ import annotations

from ..config import AgentConfig
from ..llm.base import LLMBackend
from ..memory.case import Case
from ..retrieval.ensemble import EnsembleRetriever
from ..types import MemoryDomain, Message, MessageRole

_PLAN_SYSTEM = """\
You are an expert AI agent with access to a set of tools and accumulated experience.

AVAILABLE TOOLS:
{tool_schemas}

{hints_block}{skills_block}{failure_patterns_block}PAST EXPERIENCE (retrieved from memory):
{experience}

Instructions:
- Use hints, skills, and past experience to guide your approach, but adapt to the current task.
- CRITICALLY: read the failure patterns above and AVOID those specific mistakes.
- Break the task into clear, numbered steps.
- For each step, specify: (a) what you want to achieve, (b) which tool to use, (c) what arguments.
- If a past case shows this approach failed, choose a different strategy.
- Apply any relevant hints and consolidated skills from above.
- Be concise and focused.
"""

_PLAN_USER = """\
TASK: {task}

Create a step-by-step execution plan for the task above.
Format each step as:
Step N: [thought] → tool_name(arg1=value1, arg2=value2)
"""


class Planner:
    """Retrieves relevant cases + hints and generates an execution plan."""

    def __init__(
        self,
        llm: LLMBackend,
        retriever: EnsembleRetriever,
        cfg: AgentConfig | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._cfg = cfg or AgentConfig()

    async def plan(
        self,
        task: str,
        tool_schemas: list[dict] | None = None,
        reflection: str | None = None,
        hints: str | None = None,
        failure_patterns: str | None = None,
        skills: str | None = None,
        domain: MemoryDomain | None = None,
    ) -> tuple[str, list[Case]]:
        """Return (plan_text, retrieved_cases).

        Args:
            task:             Natural-language task.
            tool_schemas:     JSON schemas of available tools.
            reflection:       Reflexion string from a previous failure (Phase 2).
            hints:            Internalized hints block from HintLibrary (Phase 4).
            failure_patterns: Anti-case patterns from FailurePatternBank (v2 CFM).
            skills:           Consolidated skill descriptions from SkillLibrary (v2 ESMC).
            domain:           Restrict retrieval to one memory domain (H-MEM pruning).
        """
        # 1. Retrieve relevant past cases. The injection gate is deliberate:
        # a weak match injected is worse than nothing for a struggling model
        # (measured), so low-scoring cases are dropped entirely.
        rcfg = self._cfg.retrieval
        results = await self._retriever.retrieve(
            task, top_k=rcfg.top_k, domain=domain
        )
        cases = [c for c, score in results if score >= rcfg.min_injection_score]

        # Phase 3.1 — plan reuse: a strong hit with a proven plan replaces
        # free-form planning entirely (no LLM call). Skipped on reflexion
        # retries: a failed attempt means the reused plan wasn't enough.
        if rcfg.enable_plan_reuse and results and reflection is None:
            top_case, top_score = results[0]
            if (
                top_score >= rcfg.plan_reuse_score
                and top_case.is_success
                and top_case.plan
            ):
                reused = (
                    f"[PLAN REUSED from a solved similar task "
                    f"(similarity {top_score:.2f})]\n"
                    "Adapt inputs/numbers to the CURRENT task; keep the structure.\n"
                    f"{top_case.plan}"
                )
                return reused, cases

        # 2. Build injected memory within a hard token budget, in priority
        # order: top case > failure patterns > skills > hints > more cases.
        # Rationale: the best exemplar carries the most signal; distilled
        # blocks are compact; extra cases are the first thing to sacrifice.
        # (~4 chars/token heuristic — exact truncation matters less than the cap.)
        budget = rcfg.injection_token_budget * 4
        guidance = [c for c in cases if c.is_success]

        def fits(text: str) -> bool:
            nonlocal budget
            if len(text) <= budget:
                budget -= len(text)
                return True
            return False

        experience_lines: list[str] = []
        failure_patterns_block = skills_block = hints_block = ""

        if guidance and fits(block := guidance[0].to_prompt_block()):
            experience_lines.append(block)
        if failure_patterns and fits(failure_patterns):
            failure_patterns_block = failure_patterns + "\n\n"
        if skills and fits(skills):
            skills_block = skills + "\n\n"
        if hints and fits(hints):
            hints_block = hints + "\n\n"
        for c in guidance[1:]:
            if fits(block := c.to_prompt_block()):
                experience_lines.append(block)

        if not experience_lines:
            experience_lines.append("No relevant successful experience found.")
        if reflection:
            experience_lines.append(
                f"\n--- REFLECTION FROM PREVIOUS FAILED ATTEMPT ---\n{reflection}\n---"
            )
        experience = "\n".join(experience_lines)

        # 3. Build prompts
        tools_text = "\n".join(
            f"- {s['name']}: {s.get('description', '')}"
            for s in (tool_schemas or [])
        )

        system = _PLAN_SYSTEM.format(
            tool_schemas=tools_text,
            experience=experience,
            hints_block=hints_block,
            failure_patterns_block=failure_patterns_block,
            skills_block=skills_block,
        )
        user_msg = _PLAN_USER.format(task=task)

        # 4. Generate plan
        resp = await self._llm.complete(
            messages=[Message(role=MessageRole.USER, content=user_msg)],
            system=system,
        )

        return resp.content.strip(), cases
