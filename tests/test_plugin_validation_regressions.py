"""Malformed caller-owned models must never poison durable memory or host results."""

import pytest
from pydantic import ValidationError

from agenticmemo import AgentMemory, ExperienceInput, KnowledgeRecord, ValidationResult

TASK = "What is the refund window?"


def malformed_experience(case):
    if case == "dict-actions":
        return {"task": TASK, "actions": "not a list"}
    if case == "dict-nested-action":
        return {"task": TASK, "actions": ["not a mapping"]}
    if case == "constructed":
        return ExperienceInput.model_construct(task=TASK, actions="not a list")
    if case == "model-copy":
        return ExperienceInput(task=TASK).model_copy(update={"actions": "not a list"})

    experience = ExperienceInput(task=TASK, actions=[{"tool": "policy_lookup"}])
    if case == "mutated-actions":
        experience.actions = "not a list"
    elif case == "mutated-task":
        experience.task = ""
    elif case == "mutated-answer":
        experience.answer = {"unexpected": "mapping"}
    elif case == "nested-action-list":
        experience.actions.append("not a mapping")
    elif case == "nested-action-key":
        experience.actions[0][1] = "keys must be strings"
    elif case == "nested-metadata-key":
        experience.metadata[1] = "keys must be strings"
    else:
        raise AssertionError(f"Unknown fixture: {case}")
    return experience


MALFORMED_CASES = [
    "dict-actions", "dict-nested-action", "constructed", "model-copy", "mutated-actions",
    "mutated-task", "mutated-answer", "nested-action-list", "nested-action-key",
    "nested-metadata-key",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MALFORMED_CASES)
async def test_malformed_experience_never_disrupts_durable_retrieval(tmp_path, case):
    path = tmp_path / "memory.sqlite"
    memory = await AgentMemory.open(path)
    try:
        await memory.ingest_knowledge([
            KnowledgeRecord(
                id="refund-policy", content="Refund window is 30 days.", source="approved-policy"
            ),
        ], scope="support")
        context = await memory.before_task(TASK, scope="support", run_id="malformed")
        with pytest.raises(ValidationError):
            await memory.after_task(
                context.handle,
                experience=malformed_experience(case),
                validation=ValidationResult.passed("policy-check", ["Policy verified"]),
            )
        assert [record.id for record in await memory.inspect(scope="support")] == ["refund-policy"]
        recalled = await memory.before_task(TASK, scope="support", run_id="after-rejection")
        assert [hit.record_id for hit in recalled.hits] == ["refund-policy"]
        assert not recalled.degraded
    finally:
        await memory.close()

    reopened = await AgentMemory.open(path)
    try:
        records = await reopened.inspect(scope="support")
        assert [record.id for record in records] == ["refund-policy"]
        recalled = await reopened.before_task(TASK, scope="support", run_id="after-restart")
        assert [hit.record_id for hit in recalled.hits] == ["refund-policy"]
        assert not recalled.degraded
    finally:
        await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MALFORMED_CASES)
async def test_adapter_preserves_host_result_when_normalized_model_is_malformed(tmp_path, case):
    memory = await AgentMemory.open(tmp_path / "memory.sqlite")
    host_result = {"task": TASK, "answer": "30 days", "opaque": object()}
    calls = {"invoke": 0, "normalize": 0, "validate": 0}

    async def invoke(task, *, extra_context):
        calls["invoke"] += 1
        return host_result

    def normalize(result):
        assert result is host_result
        calls["normalize"] += 1
        return malformed_experience(case)

    def validate(task, result):
        calls["validate"] += 1
        return ValidationResult.passed("policy-check", ["Policy verified"])

    try:
        adapter = memory.attach(invoke, normalize=normalize, validator=validate)
        wrapped = await adapter.run_with_receipt(TASK, scope="support", run_id="malformed")
        assert wrapped.result is host_result
        assert calls == {"invoke": 1, "normalize": 1, "validate": 0}
        assert wrapped.receipt.status == "error"
        assert not wrapped.receipt.durable
        assert not wrapped.receipt.eligible
        assert wrapped.context.degraded
        assert wrapped.receipt.error == "normalization failed (ValidationError)"
        assert await memory.inspect(scope="support") == []
    finally:
        await memory.close()


@pytest.mark.asyncio
async def test_valid_nested_experience_remains_independent_of_later_caller_mutation(tmp_path):
    memory = await AgentMemory.open(tmp_path / "memory.sqlite")
    actions = [{"tool": "policy_lookup", "arguments": {"departments": ["support"]}}]
    metadata = {"provenance": {"sources": ["approved-policy"]}}
    experience = ExperienceInput(task=TASK, answer="30 days", actions=actions, metadata=metadata)
    try:
        context = await memory.before_task(TASK, scope="support", run_id="valid")
        receipt = await memory.after_task(
            context.handle,
            experience=experience,
            validation=ValidationResult.passed("policy-check", ["Policy verified"]),
        )
        assert receipt.durable and receipt.eligible
        experience.actions[0]["arguments"]["departments"].append("changed")
        experience.metadata["provenance"]["sources"].clear()
        record = await memory.get(receipt.id, scope="support")
        assert record.experience.actions[0]["arguments"]["departments"] == ["support"]
        assert record.experience.metadata["provenance"]["sources"] == ["approved-policy"]
        recalled = await memory.before_task(TASK, scope="support", run_id="after-valid")
        assert [hit.record_id for hit in recalled.hits] == [receipt.id]
        assert not recalled.degraded
    finally:
        await memory.close()
