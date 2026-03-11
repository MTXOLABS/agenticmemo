"""Abstract memory backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .case import Case


class MemoryBackend(ABC):
    """All memory implementations satisfy this interface."""

    @abstractmethod
    async def store(self, case: Case) -> None:
        """Persist a case."""

    @abstractmethod
    async def get(self, case_id: str) -> Case | None:
        """Retrieve a case by id."""

    @abstractmethod
    async def delete(self, case_id: str) -> bool:
        """Delete a case. Returns True if found and deleted."""

    @abstractmethod
    async def all_cases(self) -> list[Case]:
        """Return all stored cases."""

    @abstractmethod
    async def size(self) -> int:
        """Number of cases in memory."""

    async def clear(self) -> None:
        """Delete all cases. Override for efficiency."""
        for case in await self.all_cases():
            await self.delete(case.id)
