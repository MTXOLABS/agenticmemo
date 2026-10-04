"""Escape offline demo: .venv/bin/python -m examples.plugin_memory.

The reporter is deterministic application code, not an LLM. Its validators use
fixture totals supplied separately from its output. This checks the integration
contract; it does not measure improvement on real agent tasks.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np

from agenticmemo.plugin import AgentMemory, ExperienceInput, KnowledgeRecord, ValidationResult
from agenticmemo.retrieval.embeddings import EmbeddingBackend


class LocalDemoEmbeddings(EmbeddingBackend):
    """Small deterministic token vectors; no model downloads or API calls."""

    async def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.casefold()):
                bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 64
                vectors[row, bucket] += 1.0
        return vectors


def reporter(amounts: tuple[int, ...], expected_total: int):
    """Return a host, normalizer and validator for one trusted input fixture."""

    async def invoke(task: str, *, extra_context: str):
        # The current inputs remain authoritative even when old answers appear
        # in retrieved context. The host decides how to use that context.
        total = sum(amounts)
        return {
            "task": task,
            "answer": f"Invoice total: {total}",
            "total": total,
            "amounts": list(amounts),
            "context_received": bool(extra_context),
        }

    def normalize(result):
        return ExperienceInput(
            task=result["task"],
            answer=result["answer"],
            plan="Read current invoice amounts, sum them, and check the source total.",
            actions=[{"action": "sum", "amounts": result["amounts"], "total": result["total"]}],
        )

    def validate(task: str, result):
        if result["task"] == task and result["total"] == expected_total:
            return ValidationResult.passed(
                "offline-invoice-fixture", [f"Fixture source total equals {expected_total}"]
            )
        return ValidationResult.failed("offline-invoice-fixture", "Total differs from fixture")

    return invoke, normalize, validate


async def demonstrate(path: Path) -> None:
    memory = await AgentMemory.open(path, embedder=LocalDemoEmbeddings())
    scope = "invoice-demo"
    task = "Create invoice total report"
    try:
        await memory.ingest_knowledge(
            [KnowledgeRecord(
                id="invoice-reporting", source="offline-example", version="1",
                content=("Invoice report procedure: sum current invoice amounts and state the "
                         "total. Verify the total against the source."),
            )],
            scope=scope,
        )

        # Two explicit hooks: the application controls every step between them.
        invoke, normalize, validate = reporter((10, 30), expected_total=40)
        context = await memory.before_task(task, scope=scope, run_id="direct-1")
        result = await invoke(task, extra_context=context.text)
        receipt = await memory.after_task(
            context.handle, experience=normalize(result), validation=validate(task, result)
        )
        print(f"Two hooks: total={result['total']}, hits={len(context.hits)}, "
              f"recording={receipt.status}, eligible={receipt.eligible}")

        # Adapter: same hooks, original host return value, one host invocation.
        invoke, normalize, validate = reporter((20, 45), expected_total=65)
        attached = memory.attach(invoke, normalize=normalize, validator=validate)
        wrapped = await attached.run_with_receipt(task, scope=scope, run_id="adapter-1")
        print(f"Adapter: total={wrapped.result['total']}, hits={len(wrapped.context.hits)}, "
              f"recording={wrapped.receipt.status}, eligible={wrapped.receipt.eligible}")

        # A later task uses new inputs and can retrieve the validated experience.
        invoke, normalize, validate = reporter((25, 50), expected_total=75)
        attached = memory.attach(invoke, normalize=normalize, validator=validate)
        result = await attached.run(task, scope=scope, run_id="adapter-2")
        print(f"New inputs: total={result['total']}, context_received={result['context_received']}")
    finally:
        await memory.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="agenticmemo-plugin-") as directory:
        asyncio.run(demonstrate(Path(directory) / "memory.sqlite"))
