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
from ..types import Message, MessageRole


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
    ) -> tuple[str, list[Case]]:
        """Return (plan_text, retrieved_cases).

        Args:
            task:             Natural-language task.
            tool_schemas:     JSON schemas of available tools.
            reflection:       Reflexion string from a previous failure (Phase 2).
            hints:            Internalized hints block from HintLibrary (Phase 4).
            failure_patterns: Anti-case patterns from FailurePatternBank (v2 CFM).
            skills:           Consolidated skill descriptions from SkillLibrary (v2 ESMC).
        """
        # 1. Retrieve relevant past cases
        results = await self._retriever.retrieve(
            task, top_k=self._cfg.retrieval.top_k
        )
        cases = [c for c, _ in results]

        # 2. Build experience context
        experience_lines = []
        if cases:
            for c in cases:
                experience_lines.append(c.to_prompt_block())
        else:
            experience_lines.append("No relevant past experience found.")

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
        hints_block = (hints + "\n\n") if hints else ""
        failure_patterns_block = (failure_patterns + "\n\n") if failure_patterns else ""
        skills_block = (skills + "\n\n") if skills else ""

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
