"""OpenAI LLM backend."""

from __future__ import annotations

import json
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..exceptions import LLMError
from ..types import LLMResponse, Message, MessageRole, ToolCall
from .base import LLMBackend


class OpenAILLM(LLMBackend):
    """OpenAI (GPT) backend via the `openai` SDK.

    Install extras: ``pip install agentmemento[openai]``
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed. Run: pip install agentmemento[openai]"
            ) from e
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,  # hard problems need more time
            max_retries=0,  # we handle retries via tenacity
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        try:
            import openai  # noqa: PLC0415

            openai_msgs: list[dict[str, Any]] = []
            if system:
                openai_msgs.append({"role": "system", "content": system})

            for msg in messages:
                if msg.role == MessageRole.TOOL:
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    })
                elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                    # Must include tool_calls so tool results are valid in next turn
                    openai_msgs.append({
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    })
                else:
                    openai_msgs.append({"role": msg.role.value, "content": msg.content})

            # Newer models (gpt-4.5+, o-series) require max_completion_tokens
            _uses_completion_tokens = any(
                self.model.startswith(p) for p in ("o1", "o3", "gpt-4.5", "gpt-5")
            )
            _token_key = "max_completion_tokens" if _uses_completion_tokens else "max_tokens"

            kwargs: dict[str, Any] = {
                "model": self.model,
                _token_key: self.max_tokens,
                "messages": openai_msgs,
            }
            # o-series models don't support temperature
            if not _uses_completion_tokens:
                kwargs["temperature"] = self.temperature
            if tools:
                kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
                kwargs["tool_choice"] = "auto"

            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg_out = choice.message

            text_content = msg_out.content or ""
            tool_calls: list[ToolCall] = []
            if msg_out.tool_calls:
                for tc in msg_out.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

            return LLMResponse(
                content=text_content,
                tool_calls=tool_calls,
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
                model=resp.model,
                stop_reason=choice.finish_reason or "",
            )
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            import openai  # noqa: PLC0415
            resp = await self._client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except openai.OpenAIError as e:
            raise LLMError(f"OpenAI embedding error: {e}") from e

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return tools
