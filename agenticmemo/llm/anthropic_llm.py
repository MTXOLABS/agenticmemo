"""Anthropic (Claude) LLM backend."""

from __future__ import annotations

import json
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..exceptions import LLMError
from ..types import LLMResponse, Message, MessageRole, ToolCall
from .base import LLMBackend


class AnthropicLLM(LLMBackend):
    """Claude backend via the `anthropic` SDK.

    Install extras: ``pip install agentmemento[anthropic]``
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model, temperature, max_tokens)
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install agentmemento[anthropic]"
            ) from e
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Don't retry billing/auth errors — only transient network/rate errors."""
        from ..exceptions import LLMError  # noqa: PLC0415
        if isinstance(exc, LLMError):
            msg = str(exc).lower()
            # Never retry credit / auth / invalid-request errors
            if any(k in msg for k in ("credit balance", "unauthorized", "invalid_request", "permission")):
                return False
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(lambda e: AnthropicLLM._is_retryable(e)),
    )
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        import anthropic  # noqa: PLC0415

        try:
            # Convert to Anthropic message format
            anthropic_msgs = []
            for msg in messages:
                if msg.role == MessageRole.TOOL:
                    anthropic_msgs.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }],
                    })
                elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                    # Must include tool_use blocks so tool_result is valid in next turn
                    content: list[dict[str, Any]] = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    anthropic_msgs.append({"role": "assistant", "content": content})
                else:
                    anthropic_msgs.append({
                        "role": msg.role.value,
                        "content": msg.content,
                    })

            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": anthropic_msgs,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            resp = await self._client.messages.create(**kwargs)

            # Parse response
            text_content = ""
            tool_calls: list[ToolCall] = []
            for block in resp.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    ))

            return LLMResponse(
                content=text_content,
                tool_calls=tool_calls,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model=resp.model,
                stop_reason=resp.stop_reason or "",
            )
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use sentence-transformers or OpenAI for embeddings."
        )

    def _format_tool(self, tool_schema: dict[str, Any]) -> dict[str, Any]:
        """Convert generic tool schema to Anthropic format."""
        return {
            "name": tool_schema["name"],
            "description": tool_schema.get("description", ""),
            "input_schema": tool_schema.get("parameters", {"type": "object", "properties": {}}),
        }

    def format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._format_tool(t) for t in tools]
