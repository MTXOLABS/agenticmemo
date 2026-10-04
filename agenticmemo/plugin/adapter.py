"""A narrow adapter that adds memory without owning the host agent's runtime."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from .models import (
    ContextHandle,
    ExperienceInput,
    MemoryConfigurationError,
    MemoryContext,
    RecordingReceipt,
    ValidationResult,
)

if TYPE_CHECKING:
    from .memory import AgentMemory


class WrappedResult(BaseModel):
    """Per-call diagnostics alongside the unchanged host result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: Any
    receipt: RecordingReceipt
    context: MemoryContext


async def _call(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call exactly once, keeping synchronous application work off the event loop."""
    is_async = inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )
    if is_async:
        value = callback(*args, **kwargs)
    else:
        value = await asyncio.to_thread(callback, *args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


def _degraded(context: MemoryContext, message: str) -> MemoryContext:
    return context.model_copy(
        update={"degraded": True, "diagnostics": [*context.diagnostics, message]}
    )


class MemoryAgent:
    """Attach memory to an application-owned callable.

    ``invoke(task, extra_context=..., **kwargs)`` owns execution, tools and retries.
    ``normalize(result)`` selects the experience safe to persist. The optional
    ``validator(task, result)`` independently checks the result. Each callback may
    be synchronous or asynchronous. No callback is retried by this adapter.
    """

    def __init__(
        self,
        memory: AgentMemory,
        invoke: Callable[..., Any],
        normalize: Callable[[Any], Any],
        validator: Callable[[str, Any], Any] | None = None,
        *,
        validation_timeout: float = 10.0,
    ) -> None:
        if not callable(invoke) or not callable(normalize):
            raise MemoryConfigurationError("invoke and normalize must be callable")
        if validator is not None and not callable(validator):
            raise MemoryConfigurationError("validator must be callable")
        if (
            isinstance(validation_timeout, bool)
            or not isinstance(validation_timeout, (int, float))
            or not math.isfinite(validation_timeout)
            or validation_timeout <= 0
        ):
            raise MemoryConfigurationError("validation_timeout must be a positive finite number")
        self._memory = memory
        self._invoke = invoke
        self._normalize = normalize
        self._validator = validator
        self._validation_timeout = validation_timeout

    async def run(
        self,
        task: str,
        *,
        scope: str,
        run_id: str,
        token_budget: int = 1200,
        **kwargs: Any,
    ) -> Any:
        """Return the original host result, including when memory fails soft."""
        wrapped = await self.run_with_receipt(
            task, scope=scope, run_id=run_id, token_budget=token_budget, **kwargs
        )
        return wrapped.result

    async def __call__(
        self,
        task: str,
        *,
        scope: str,
        run_id: str,
        token_budget: int = 1200,
        **kwargs: Any,
    ) -> Any:
        return await self.run(
            task, scope=scope, run_id=run_id, token_budget=token_budget, **kwargs
        )

    async def run_with_receipt(
        self,
        task: str,
        *,
        scope: str,
        run_id: str,
        token_budget: int = 1200,
        **kwargs: Any,
    ) -> WrappedResult:
        """Return an independent result/context/receipt for this invocation.

        Host exceptions and cancellation propagate. Invalid integration inputs
        and memory configuration errors also propagate; other memory failures
        degrade the per-call context. Validation failure produces an unknown
        verdict, which is ineligible for experience retrieval.
        """
        self._validate_inputs(task, scope, run_id, token_budget, kwargs)
        try:
            context = await self._memory.before_task(
                task, scope=scope, run_id=run_id, token_budget=token_budget
            )
        except MemoryConfigurationError:
            raise
        except Exception as exc:
            context = MemoryContext(
                handle=ContextHandle(task=task, scope=scope, run_id=run_id),
                degraded=True,
                diagnostics=[f"before_task failed ({type(exc).__name__})"],
            )
        if (
            context.handle.scope != scope
            or context.handle.run_id != run_id
            or context.handle.task != task
        ):
            raise MemoryConfigurationError("memory context handle does not match this invocation")

        # Deliberately outside the memory exception handlers: the host owns its
        # own failures and retries, and must not be invoked a second time here.
        result = await _call(self._invoke, task, extra_context=context.text, **kwargs)

        try:
            experience = ExperienceInput.model_validate(await _call(self._normalize, result))
            if experience.task != task:
                raise ValueError("normalized experience task does not match the invocation")
        except Exception as exc:
            message = f"normalization failed ({type(exc).__name__})"
            return WrappedResult(
                result=result,
                context=_degraded(context, message),
                receipt=RecordingReceipt(status="error", error=message),
            )

        validation = None
        if self._validator is not None:
            try:
                verdict = await asyncio.wait_for(
                    _call(self._validator, task, result), timeout=self._validation_timeout
                )
                validation = ValidationResult.model_validate(verdict)
            except Exception as exc:
                message = f"validation failed ({type(exc).__name__})"
                validation = ValidationResult.unknown(message)
                context = _degraded(context, message)

        try:
            receipt = await self._memory.after_task(
                context.handle, experience=experience, validation=validation
            )
        except MemoryConfigurationError:
            raise
        except Exception as exc:
            message = f"after_task failed ({type(exc).__name__})"
            receipt = RecordingReceipt(status="error", error=message, retryable=True)
            context = _degraded(context, message)
        else:
            if receipt.status == "error":
                context = _degraded(context, "memory recording returned an error receipt")
        return WrappedResult(result=result, receipt=receipt, context=context)

    @staticmethod
    def _validate_inputs(
        task: str,
        scope: str,
        run_id: str,
        token_budget: int,
        kwargs: dict[str, Any],
    ) -> None:
        for label, value in (("task", task), ("scope", scope), ("run_id", run_id)):
            if not isinstance(value, str) or not value.strip():
                raise MemoryConfigurationError(f"{label} must be a nonblank string")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0:
            raise MemoryConfigurationError("token_budget must be a nonnegative integer")
        if "extra_context" in kwargs:
            raise MemoryConfigurationError("extra_context is supplied by the memory adapter")
