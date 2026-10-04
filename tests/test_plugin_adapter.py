"""Offline checks for the adapter's host-runtime and validation boundaries."""

import asyncio
import threading

import pytest

from agenticmemo.plugin.adapter import MemoryAgent
from agenticmemo.plugin.memory import AgentMemory
from agenticmemo.plugin.models import (
    ContextHandle,
    ExperienceInput,
    KnowledgeRecord,
    MemoryConfigurationError,
    MemoryContext,
    RecordingReceipt,
    ValidationResult,
)


class _Memory:
    def __init__(self, before_error=None, after_error=None):
        self.before_error = before_error
        self.after_error = after_error
        self.before_calls: list[tuple[str, str, str, int]] = []
        self.after_calls: list[tuple[ContextHandle, ExperienceInput, ValidationResult | None]] = []

    async def before_task(self, task, *, scope, run_id, token_budget):
        self.before_calls.append((task, scope, run_id, token_budget))
        if self.before_error is not None:
            raise self.before_error
        return MemoryContext(
            text=f"context for {task}",
            handle=ContextHandle(task=task, scope=scope, run_id=run_id),
        )

    async def after_task(self, handle, *, experience, validation):
        self.after_calls.append((handle, experience, validation))
        if self.after_error is not None:
            raise self.after_error
        return RecordingReceipt(
            id=handle.run_id,
            status="stored",
            durable=True,
            indexed=True,
            eligible=validation is not None and validation.status == "passed",
        )


def _normalize(result):
    return ExperienceInput(task=result["task"], answer=result["answer"])


@pytest.mark.asyncio
async def test_async_callbacks_preserve_identity_and_invoke_host_once():
    memory = _Memory()
    original = {"task": "report", "answer": "42"}
    calls = []

    async def invoke(task, *, extra_context, mode):
        calls.append((task, extra_context, mode))
        return original

    async def normalize(result):
        assert result is original
        return _normalize(result)

    async def validate(task, result):
        assert task == "report" and result is original
        return ValidationResult.passed("report-check", ["Total equals 42"])

    wrapped = await MemoryAgent(memory, invoke, normalize, validate).run_with_receipt(
        "report", scope="team-a", run_id="r1", token_budget=200, mode="preview"
    )
    assert wrapped.result is original
    assert calls == [("report", "context for report", "preview")]
    assert memory.before_calls == [("report", "team-a", "r1", 200)]
    assert len(memory.after_calls) == 1
    assert wrapped.receipt.eligible and not wrapped.context.degraded


@pytest.mark.asyncio
async def test_run_and_callable_return_original_result_without_shared_receipt():
    original = {"task": "task", "answer": "answer"}

    async def invoke(task, *, extra_context):
        return original

    adapter = MemoryAgent(_Memory(), invoke, _normalize)
    assert await adapter.run("task", scope="s", run_id="one") is original
    assert await adapter("task", scope="s", run_id="two") is original
    assert not hasattr(adapter, "last_receipt")


@pytest.mark.asyncio
async def test_sync_callbacks_run_off_event_loop_and_keep_original_result():
    thread_ids = []
    original = {"task": "task", "answer": "answer"}

    def invoke(task, *, extra_context):
        thread_ids.append(threading.get_ident())
        return original

    def normalize(result):
        thread_ids.append(threading.get_ident())
        return _normalize(result)

    def validate(task, result):
        thread_ids.append(threading.get_ident())
        return ValidationResult.passed("check", ["checked artifact"])

    wrapped = await MemoryAgent(_Memory(), invoke, normalize, validate).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert wrapped.result is original and wrapped.receipt.eligible
    assert len(thread_ids) == 3
    assert all(thread_id != threading.get_ident() for thread_id in thread_ids)


@pytest.mark.asyncio
async def test_sync_callback_returning_awaitable_is_resolved_once():
    calls = []

    def invoke(task, *, extra_context):
        calls.append(task)

        async def result():
            return {"task": task, "answer": "answer"}

        return result()

    result = await MemoryAgent(_Memory(), invoke, _normalize).run(
        "task", scope="s", run_id="r"
    )
    assert result["answer"] == "answer" and calls == ["task"]


@pytest.mark.asyncio
async def test_missing_validator_records_without_positive_verdict():
    memory = _Memory()

    async def invoke(task, *, extra_context):
        return {"task": task, "answer": "success according to host"}

    wrapped = await MemoryAgent(memory, invoke, _normalize).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert memory.after_calls[0][2] is None
    assert not wrapped.receipt.eligible
    assert not wrapped.context.degraded


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["raise", "invalid", "wrong_task"])
async def test_bad_normalization_keeps_host_result_and_does_not_record(problem):
    memory = _Memory()
    original = {"task": "task", "answer": "answer"}

    async def invoke(task, *, extra_context):
        return original

    async def normalize(result):
        if problem == "raise":
            raise RuntimeError("private host data must not appear in diagnostics")
        if problem == "invalid":
            return {"answer": "missing task"}
        return ExperienceInput(task="other task")

    async def validator(task, result):
        pytest.fail("validation must not run for an invalid experience")

    wrapped = await MemoryAgent(memory, invoke, normalize, validator).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert wrapped.result is original and wrapped.receipt.status == "error"
    assert wrapped.context.degraded and not memory.after_calls
    assert "private host data" not in str(wrapped.context.diagnostics)
    assert "private host data" not in wrapped.receipt.error


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["raise", "invalid", "unproven_pass"])
async def test_bad_validator_is_unknown_and_ineligible(problem):
    memory = _Memory()

    async def invoke(task, *, extra_context):
        return {"task": task, "answer": "answer"}

    async def validate(task, result):
        if problem == "raise":
            raise RuntimeError("sensitive validation payload")
        if problem == "invalid":
            return {"status": "success"}
        return {"status": "passed", "validator": "", "evidence": []}

    wrapped = await MemoryAgent(memory, invoke, _normalize, validate).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert wrapped.result["answer"] == "answer"
    assert memory.after_calls[0][2].status == "unknown"
    assert wrapped.context.degraded and not wrapped.receipt.eligible
    assert "sensitive validation payload" not in str(wrapped.context.diagnostics)


@pytest.mark.asyncio
async def test_validation_timeout_records_unknown_and_cancels_async_validator():
    memory = _Memory()
    cancelled = asyncio.Event()

    async def invoke(task, *, extra_context):
        return {"task": task, "answer": "answer"}

    async def validate(task, result):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    wrapped = await MemoryAgent(
        memory, invoke, _normalize, validate, validation_timeout=0.01
    ).run_with_receipt("task", scope="s", run_id="r")
    assert cancelled.is_set()
    assert memory.after_calls[0][2].status == "unknown"
    assert not wrapped.receipt.eligible
    assert wrapped.context.diagnostics == ["validation failed (TimeoutError)"]


@pytest.mark.asyncio
async def test_unexpected_hook_errors_fail_soft_without_retry():
    memory = _Memory(before_error=OSError("private path"), after_error=OSError("private path"))
    calls = []

    async def invoke(task, *, extra_context):
        calls.append(extra_context)
        return {"task": task, "answer": "answer"}

    wrapped = await MemoryAgent(memory, invoke, _normalize).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert calls == [""] and len(memory.after_calls) == 1
    assert memory.after_calls[0][0] == ContextHandle(task="task", scope="s", run_id="r")
    assert wrapped.result["answer"] == "answer"
    assert wrapped.receipt.status == "error" and wrapped.receipt.retryable
    assert wrapped.context.diagnostics == [
        "before_task failed (OSError)", "after_task failed (OSError)"
    ]
    assert "private path" not in wrapped.receipt.error


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "after"])
async def test_memory_configuration_error_propagates(phase):
    error = MemoryConfigurationError("invalid integration")
    memory = _Memory(
        before_error=error if phase == "before" else None,
        after_error=error if phase == "after" else None,
    )
    calls = []

    async def invoke(task, *, extra_context):
        calls.append(task)
        return {"task": task, "answer": "answer"}

    with pytest.raises(MemoryConfigurationError):
        await MemoryAgent(memory, invoke, _normalize).run("task", scope="s", run_id="r")
    assert len(calls) == (0 if phase == "before" else 1)


@pytest.mark.asyncio
async def test_host_failure_propagates_without_normalization_recording_or_retry():
    memory = _Memory()
    calls = []
    error = RuntimeError("host failure")

    async def invoke(task, *, extra_context):
        calls.append(task)
        raise error

    with pytest.raises(RuntimeError) as raised:
        await MemoryAgent(memory, invoke, _normalize).run("task", scope="s", run_id="r")
    assert raised.value is error
    assert calls == ["task"] and not memory.after_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before", "invoke", "normalize", "validate", "after"])
async def test_cancellation_is_never_converted_to_a_receipt(phase):
    memory = _Memory(
        before_error=asyncio.CancelledError() if phase == "before" else None,
        after_error=asyncio.CancelledError() if phase == "after" else None,
    )

    async def invoke(task, *, extra_context):
        if phase == "invoke":
            raise asyncio.CancelledError()
        return {"task": task, "answer": "answer"}

    async def normalize(result):
        if phase == "normalize":
            raise asyncio.CancelledError()
        return _normalize(result)

    async def validate(task, result):
        if phase == "validate":
            raise asyncio.CancelledError()
        return ValidationResult.unknown()

    with pytest.raises(asyncio.CancelledError):
        await MemoryAgent(memory, invoke, normalize, validate).run(
            "task", scope="s", run_id="r"
        )
    assert len(memory.after_calls) == (1 if phase == "after" else 0)


@pytest.mark.asyncio
async def test_mismatched_handle_fails_closed_before_host_execution():
    class WrongScope(_Memory):
        async def before_task(self, task, **kwargs):
            return MemoryContext(handle=ContextHandle(task=task, scope="other", run_id="r"))

    async def invoke(task, *, extra_context):
        pytest.fail("host must not be invoked with another scope's memory")

    with pytest.raises(MemoryConfigurationError, match="handle"):
        await MemoryAgent(WrongScope(), invoke, _normalize).run("task", scope="s", run_id="r")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"task": " "}, {"scope": ""}, {"run_id": ""}, {"token_budget": -1},
        {"token_budget": True}, {"token_budget": 1.5}, {"extra_context": "caller context"},
    ],
)
async def test_invalid_inputs_fail_before_memory_or_host(overrides):
    memory = _Memory()

    async def invoke(task, *, extra_context):
        pytest.fail("invalid input must not invoke host")

    arguments = {"task": "task", "scope": "s", "run_id": "r", **overrides}
    with pytest.raises(MemoryConfigurationError):
        await MemoryAgent(memory, invoke, _normalize).run(**arguments)
    assert not memory.before_calls and not memory.after_calls


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_validation_timeout_fails_at_attachment(timeout):
    with pytest.raises(MemoryConfigurationError, match="validation_timeout"):
        MemoryAgent(_Memory(), lambda task, **kwargs: None, _normalize, validation_timeout=timeout)


@pytest.mark.asyncio
async def test_concurrent_runs_keep_context_and_receipts_separate():
    memory = _Memory()
    both_started = asyncio.Event()
    calls: list[tuple[str, str]] = []

    async def invoke(task, *, extra_context):
        calls.append((task, extra_context))
        if len(calls) == 2:
            both_started.set()
        await both_started.wait()
        return {"task": task, "answer": task}

    adapter = MemoryAgent(memory, invoke, _normalize)
    first, second = await asyncio.gather(
        adapter.run_with_receipt("first", scope="one", run_id="r1"),
        adapter.run_with_receipt("second", scope="two", run_id="r2"),
    )
    assert calls == [("first", "context for first"), ("second", "context for second")]
    assert first.result["task"] == first.context.handle.task == "first"
    assert second.result["task"] == second.context.handle.task == "second"
    assert first.receipt.id == "r1" and second.receipt.id == "r2"
    assert first.context.handle.scope == "one" and second.context.handle.scope == "two"


@pytest.mark.asyncio
async def test_error_receipt_is_visible_in_context_without_changing_result():
    class ErrorReceipt(_Memory):
        async def after_task(self, handle, *, experience, validation):
            return RecordingReceipt(status="error", error="storage unavailable", retryable=True)

    async def invoke(task, *, extra_context):
        return {"task": task, "answer": "answer"}

    wrapped = await MemoryAgent(ErrorReceipt(), invoke, _normalize).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert wrapped.result["answer"] == "answer" and wrapped.context.degraded
    assert wrapped.receipt.error == "storage unavailable"


@pytest.mark.asyncio
async def test_failed_validation_preserves_failure_evidence():
    memory = _Memory()

    async def invoke(task, *, extra_context):
        return {"task": task, "answer": "incorrect"}

    async def validate(task, result):
        return ValidationResult.failed("total-check", "Expected 42", ["Observed incorrect"])

    wrapped = await MemoryAgent(memory, invoke, _normalize, validate).run_with_receipt(
        "task", scope="s", run_id="r"
    )
    assert memory.after_calls[0][2].status == "failed"
    assert memory.after_calls[0][2].evidence == ["Observed incorrect"]
    assert not wrapped.receipt.eligible and not wrapped.context.degraded


@pytest.mark.asyncio
async def test_factory_adapter_uses_real_store_and_retrieves_validated_experience(tmp_path):
    memory = await AgentMemory.open(tmp_path / "memory.sqlite")
    original = {"task": "Create invoice total report", "answer": "42", "total": 42}
    host_calls = []

    async def invoke(task, *, extra_context):
        host_calls.append(extra_context)
        return original

    def validate(task, result):
        return ValidationResult.passed("trusted-fixture", ["Source total equals 42"])

    try:
        await memory.ingest_knowledge(
            [KnowledgeRecord(
                id="invoice-guide", source="approved-guide",
                content="Invoice total report: sum the current invoice amounts.",
            )],
            scope="reports",
        )
        adapter = memory.attach(invoke, normalize=_normalize, validator=validate)
        first = await adapter.run_with_receipt(original["task"], scope="reports", run_id="r1")
        second = await adapter.run_with_receipt(original["task"], scope="reports", run_id="r2")
        assert first.result is original and second.result is original
        assert len(host_calls) == 2
        assert first.receipt.durable and first.receipt.eligible
        assert second.receipt.durable and second.receipt.eligible
        assert any(hit.kind == "experience" for hit in second.context.hits)
        stored = await memory.get(first.receipt.id, scope="reports")
        assert stored.experience.answer == "42" and stored.validation.status == "passed"
        assert await memory.get(first.receipt.id, scope="other") is None
    finally:
        await memory.close()
