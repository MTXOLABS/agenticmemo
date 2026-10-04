"""Conservative scope-level provenance for reference-dependent experience."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .models import MemoryRecord


def knowledge_revision(records: Iterable[MemoryRecord]) -> str:
    """Fingerprint reference state, including updates, deletions and expiry.

    A scope is the dependency boundary: an experience may depend indirectly on
    any of its references through earlier experience. Tracking the full scope
    avoids presenting old answers as current after an upstream policy changes.
    """
    references = [
        {
            "id": record.id,
            "knowledge": record.knowledge.model_dump(mode="json"),
            "updated_at": record.updated_at.isoformat(),
            "eligible": record.eligible,
        }
        for record in records if record.knowledge is not None
    ]
    payload = json.dumps(
        sorted(references, key=lambda item: item["id"]),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
