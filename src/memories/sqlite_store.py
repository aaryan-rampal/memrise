from __future__ import annotations

import sqlite3
import time
import uuid
from json import dumps, loads
from pathlib import Path
from typing import Any

from memories.embedder import Embedding
from memories.models import AddMemory, Memory, MemoryType, Source, validate_importance


class SQLiteMemoryStore:
    """SQLite-backed memory storage."""

    def __init__(self, db_path: Path | str) -> None:
        """Create a SQLite store for one database file."""
        self._db_path = Path(db_path)

    def initialize(self) -> None:
        """Create database schema if it does not already exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                USING fts5(id UNINDEXED, content);

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                """
            )

    def add(self, memory: AddMemory) -> Memory:
        """Persist a memory and index it for FTS search."""
        now = int(time.time())
        stored = Memory(
            id=uuid.uuid4().hex,
            content=memory.content,
            source=Source.parse(memory.source),
            memory_type=MemoryType.parse(memory.memory_type),
            importance=memory.importance,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories
                  (id, content, source, memory_type, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                self._memory_values(stored),
            )
            connection.execute(
                "INSERT INTO memory_fts (id, content) VALUES (?, ?)",
                (stored.id, stored.content),
            )
        return stored

    def save_embedding(self, memory_id: str, embedding: Embedding) -> None:
        """Persist one embedding for a memory."""
        if self.get(memory_id) is None:
            msg = f"Memory '{memory_id}' does not exist"
            raise ValueError(msg)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_embeddings
                  (memory_id, provider, model, dimensions, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                  provider = excluded.provider,
                  model = excluded.model,
                  dimensions = excluded.dimensions,
                  vector_json = excluded.vector_json,
                  created_at = excluded.created_at
                """,
                (
                    memory_id,
                    embedding.provider,
                    embedding.model,
                    embedding.dimensions,
                    dumps(embedding.vector),
                    int(time.time()),
                ),
            )

    def get_embedding(self, memory_id: str) -> Embedding | None:
        """Return one stored embedding by memory id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        return Embedding(
            provider=row["provider"],
            model=row["model"],
            vector=[float(value) for value in loads(row["vector_json"])],
        )

    def get(self, memory_id: str) -> Memory | None:
        """Return one memory by id, if it exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        return self._memory_from_row(row)

    def recent(self, limit: int) -> list[Memory]:
        """Return newest memories first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def search(self, query: str, limit: int) -> list[Memory]:
        """Search memory content using SQLite FTS5."""
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*
                FROM memory_fts f
                JOIN memories m ON m.id = f.id
                WHERE memory_fts MATCH ?
                ORDER BY bm25(memory_fts), m.updated_at DESC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        source: str | None = None,
        memory_type: str | None = None,
        importance: float | None = None,
    ) -> Memory:
        """Update a memory and keep its search index in sync."""
        existing = self.get(memory_id)
        if existing is None:
            msg = f"Memory '{memory_id}' does not exist"
            raise ValueError(msg)
        updated = self._updated_memory(existing, content, source, memory_type, importance)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE memories
                SET content = ?, source = ?, memory_type = ?, importance = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    updated.content,
                    updated.source.value,
                    updated.memory_type.value,
                    updated.importance,
                    updated.updated_at,
                    updated.id,
                ),
            )
            connection.execute("DELETE FROM memory_fts WHERE id = ?", (updated.id,))
            connection.execute(
                "INSERT INTO memory_fts (id, content) VALUES (?, ?)",
                (updated.id, updated.content),
            )
        return updated

    def delete(self, memory_id: str) -> bool:
        """Delete a memory and its search index row."""
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            connection.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _updated_memory(
        self,
        existing: Memory,
        content: str | None,
        source: str | None,
        memory_type: str | None,
        importance: float | None,
    ) -> Memory:
        return Memory(
            id=existing.id,
            content=self._content_value(content, existing.content),
            source=Source.parse(source) if source is not None else existing.source,
            memory_type=(
                MemoryType.parse(memory_type) if memory_type is not None else existing.memory_type
            ),
            importance=(
                validate_importance(importance) if importance is not None else existing.importance
            ),
            created_at=existing.created_at,
            updated_at=int(time.time()),
        )

    @staticmethod
    def _content_value(content: str | None, fallback: str) -> str:
        if content is None:
            return fallback
        stripped = content.strip()
        if not stripped:
            msg = "Memory content cannot be empty"
            raise ValueError(msg)
        return stripped

    @staticmethod
    def _memory_values(memory: Memory) -> tuple[str, str, str, str, float, int, int]:
        return (
            memory.id,
            memory.content,
            memory.source.value,
            memory.memory_type.value,
            memory.importance,
            memory.created_at,
            memory.updated_at,
        )

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> Memory:
        row_data: dict[str, Any] = dict(row)
        return Memory(
            id=row_data["id"],
            content=row_data["content"],
            source=Source.parse(row_data["source"]),
            memory_type=MemoryType.parse(row_data["memory_type"]),
            importance=row_data["importance"],
            created_at=row_data["created_at"],
            updated_at=row_data["updated_at"],
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [term.strip('"') for term in query.split() if term.strip('"')]
        return " ".join(f'"{term}"' for term in terms)
