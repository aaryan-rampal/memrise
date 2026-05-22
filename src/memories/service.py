from __future__ import annotations

from memories.embedder import Embedder
from memories.models import AddMemory, Memory
from memories.store import MemoryStore


class MemoryService:
    """Application service for memory workflows."""

    def __init__(self, store: MemoryStore, *, embedder: Embedder | None = None) -> None:
        """Create the service with a storage backend."""
        self._store = store
        self._embedder = embedder

    def add_memory(self, memory: AddMemory) -> Memory:
        """Create a memory."""
        stored = self._store.add(memory)
        if self._embedder is not None:
            self._store.save_embedding(stored.id, self._embedder.embed(stored.content))
        return stored

    def recent(self, limit: int) -> list[Memory]:
        """Return recent memories."""
        return self._store.recent(limit)

    def search(self, query: str, limit: int) -> list[Memory]:
        """Search memory content."""
        return self._store.search(query, limit)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        source: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
    ) -> Memory:
        """Update a memory."""
        return self._store.update(
            memory_id,
            content=content,
            source=source,
            memory_type=memory_type,
            importance=importance,
        )

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        return self._store.delete(memory_id)
