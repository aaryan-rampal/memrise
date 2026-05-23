from __future__ import annotations

from typing import Protocol

from memories.embedder import Embedding
from memories.models import AddMemory, Memory


class MemoryStore(Protocol):
    """Persistence contract for memory storage backends."""

    def add(self, memory: AddMemory) -> Memory:
        """Persist a memory and return the stored record."""

    def save_embedding(self, memory_id: str, embedding: Embedding) -> None:
        """Persist an embedding for a memory."""

    def get_embedding(self, memory_id: str) -> Embedding | None:
        """Return one memory embedding, if it exists."""

    def get(self, memory_id: str) -> Memory | None:
        """Return one memory by id, if it exists."""

    def recent(self, limit: int) -> list[Memory]:
        """Return recent memories in newest-first order."""

    def search(self, query: str, limit: int) -> list[Memory]:
        """Search memories and return best matches first."""

    def semantic_search(self, query_embedding: Embedding, limit: int) -> list[tuple[Memory, float]]:
        """Search memories by embedding similarity."""

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        source: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
    ) -> Memory:
        """Update one memory and return the updated record."""

    def delete(self, memory_id: str) -> bool:
        """Delete one memory by id."""
