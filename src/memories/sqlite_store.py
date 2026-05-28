from __future__ import annotations

import sqlite3
import time
import uuid
from json import dumps, loads
from math import sqrt
from pathlib import Path
from typing import Any

from memories.embedder import Embedding
from memories.models import (
    AddMemory,
    Memory,
    MemoryType,
    RawArtifact,
    RawArtifactSearchMatch,
    RawArtifactSpan,
    Source,
    validate_importance,
)


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

                CREATE TABLE IF NOT EXISTS raw_artifacts (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_conversation_id TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT,
                    title TEXT,
                    workspace TEXT,
                    content TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS raw_artifact_spans (
                    id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    span_index INTEGER NOT NULL,
                    message_index INTEGER NOT NULL,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (artifact_id) REFERENCES raw_artifacts(id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS raw_artifact_span_fts
                USING fts5(span_id UNINDEXED, content);

                CREATE TABLE IF NOT EXISTS raw_artifact_embeddings (
                    artifact_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (artifact_id) REFERENCES raw_artifacts(id) ON DELETE CASCADE
                );
                """
            )
            self._recreate_raw_schema_if_legacy(connection)
            self._ensure_raw_schema(connection)

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

    def upsert_memories(self, memories: list[Memory]) -> None:
        """Persist memories with caller-provided stable ids and refresh FTS rows."""
        if not memories:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO memories
                  (id, content, source, memory_type, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  content = excluded.content,
                  source = excluded.source,
                  memory_type = excluded.memory_type,
                  importance = excluded.importance,
                  updated_at = excluded.updated_at
                """,
                [self._memory_values(memory) for memory in memories],
            )
            connection.executemany(
                "DELETE FROM memory_fts WHERE id = ?",
                [(memory.id,) for memory in memories],
            )
            connection.executemany(
                "INSERT INTO memory_fts (id, content) VALUES (?, ?)",
                [(memory.id, memory.content) for memory in memories],
            )

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

    def semantic_search(self, query_embedding: Embedding, limit: int) -> list[tuple[Memory, float]]:
        """Search memory embeddings by exact cosine similarity."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*, e.vector_json
                FROM memory_embeddings e
                JOIN memories m ON m.id = e.memory_id
                """
            ).fetchall()
        scored = [
            (
                self._memory_from_row(row),
                self._cosine_similarity(query_embedding.vector, self._vector_from_row(row)),
            )
            for row in rows
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def upsert_raw_artifact(
        self,
        artifact: RawArtifact,
        spans: list[RawArtifactSpan],
    ) -> None:
        """Persist a raw chat artifact and index it for FTS search."""
        self.upsert_raw_artifacts([artifact], spans)

    def upsert_raw_artifacts(
        self,
        artifacts: list[RawArtifact],
        spans: list[RawArtifactSpan],
    ) -> None:
        """Persist raw chat artifacts and index them for FTS search."""
        if not artifacts:
            return
        for artifact in artifacts:
            self._validate_raw_artifact(artifact)
        for span in spans:
            self._validate_raw_artifact_span(span)
        values = [self._raw_artifact_values(artifact) for artifact in artifacts]
        span_values = [self._raw_artifact_span_values(span) for span in spans]
        fts_values = [(span.id, span.content) for span in spans]
        artifact_ids = [(artifact.id,) for artifact in artifacts]
        with self._connect() as connection:
            connection.executemany(
                """
                DELETE FROM raw_artifact_span_fts
                WHERE span_id IN (
                  SELECT id FROM raw_artifact_spans WHERE artifact_id = ?
                )
                """,
                artifact_ids,
            )
            connection.executemany(
                "DELETE FROM raw_artifact_spans WHERE artifact_id = ?",
                artifact_ids,
            )
            connection.executemany(
                """
                INSERT INTO raw_artifacts
                  (
                    id,
                    provider,
                    source_path,
                    source_conversation_id,
                    created_at,
                    updated_at,
                    title,
                    workspace,
                    content
                  )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  provider = excluded.provider,
                  source_path = excluded.source_path,
                  source_conversation_id = excluded.source_conversation_id,
                  created_at = excluded.created_at,
                  updated_at = excluded.updated_at,
                  title = excluded.title,
                  workspace = excluded.workspace,
                  content = excluded.content
                """,
                values,
            )
            connection.executemany(
                """
                INSERT INTO raw_artifact_spans
                  (
                    id,
                    artifact_id,
                    span_index,
                    message_index,
                    message_id,
                    role,
                    created_at,
                    start_offset,
                    end_offset,
                    content
                  )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                span_values,
            )
            connection.executemany(
                "INSERT INTO raw_artifact_span_fts (span_id, content) VALUES (?, ?)",
                fts_values,
            )

    def replace_raw_artifacts(
        self,
        artifacts: list[RawArtifact],
        spans: list[RawArtifactSpan],
        providers: list[str] | None,
    ) -> None:
        """Replace indexed raw artifacts for the selected providers."""
        for artifact in artifacts:
            self._validate_raw_artifact(artifact)
        for span in spans:
            self._validate_raw_artifact_span(span)
        with self._connect() as connection:
            self._delete_stale_raw_artifacts(connection, artifacts, providers)
            if artifacts:
                connection.executemany(
                    """
                    INSERT INTO raw_artifacts
                      (
                        id,
                        provider,
                        source_path,
                        source_conversation_id,
                        created_at,
                        updated_at,
                        title,
                        workspace,
                        content
                      )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      provider = excluded.provider,
                      source_path = excluded.source_path,
                      source_conversation_id = excluded.source_conversation_id,
                      created_at = excluded.created_at,
                      updated_at = excluded.updated_at,
                      title = excluded.title,
                      workspace = excluded.workspace,
                      content = excluded.content
                    """,
                    [self._raw_artifact_values(artifact) for artifact in artifacts],
                )
            connection.execute("DELETE FROM raw_artifact_spans")
            if spans:
                connection.executemany(
                    """
                    INSERT INTO raw_artifact_spans
                      (
                        id,
                        artifact_id,
                        span_index,
                        message_index,
                        message_id,
                        role,
                        created_at,
                        start_offset,
                        end_offset,
                        content
                      )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [self._raw_artifact_span_values(span) for span in spans],
                )
            self._rebuild_raw_artifact_span_fts(connection)

    def search_raw_artifacts(self, query: str, limit: int) -> list[RawArtifactSearchMatch]:
        """Search raw chat artifacts using SQLite FTS5."""
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, s.id AS span_id, s.artifact_id, s.span_index, s.message_index,
                  s.message_id, s.role, s.created_at AS span_created_at, s.start_offset,
                  s.end_offset, s.content AS span_content
                FROM raw_artifact_span_fts f
                JOIN raw_artifact_spans s ON s.id = f.span_id
                JOIN raw_artifacts r ON r.id = s.artifact_id
                WHERE raw_artifact_span_fts MATCH ?
                ORDER BY bm25(raw_artifact_span_fts), s.span_index
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [self._raw_artifact_search_match_from_row(row) for row in rows]

    def raw_artifact_count(self) -> int:
        """Return the number of indexed raw artifacts."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM raw_artifacts").fetchone()
        return int(row["count"])

    def save_raw_artifact_embedding(self, artifact_id: str, embedding: Embedding) -> None:
        """Persist one embedding for a raw artifact."""
        self.save_raw_artifact_embeddings([(artifact_id, embedding)])

    def save_raw_artifact_embeddings(
        self,
        embeddings: list[tuple[str, Embedding]],
    ) -> None:
        """Persist embeddings for raw artifacts."""
        if not embeddings:
            return
        with self._connect() as connection:
            missing = self._missing_raw_artifact_ids(connection, embeddings)
            if missing:
                msg = f"Raw artifact '{missing[0]}' does not exist"
                raise ValueError(msg)
            connection.executemany(
                """
                INSERT INTO raw_artifact_embeddings
                  (artifact_id, provider, model, dimensions, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                  provider = excluded.provider,
                  model = excluded.model,
                  dimensions = excluded.dimensions,
                  vector_json = excluded.vector_json,
                  created_at = excluded.created_at
                """,
                [
                    (
                        artifact_id,
                        embedding.provider,
                        embedding.model,
                        embedding.dimensions,
                        dumps(embedding.vector),
                        int(time.time()),
                    )
                    for artifact_id, embedding in embeddings
                ],
            )

    def clear_raw_artifact_embeddings(self, providers: list[str] | None = None) -> None:
        """Delete raw artifact embeddings for the selected providers."""
        with self._connect() as connection:
            if providers is None:
                connection.execute("DELETE FROM raw_artifact_embeddings")
                return
            for provider in providers:
                connection.execute(
                    """
                    DELETE FROM raw_artifact_embeddings
                    WHERE artifact_id IN (
                      SELECT id FROM raw_artifacts WHERE provider = ?
                    )
                    """,
                    (provider,),
                )

    def raw_artifact_embedding_count(self) -> int:
        """Return the number of raw artifact embeddings."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM raw_artifact_embeddings"
            ).fetchone()
        return int(row["count"])

    def raw_artifact_ids_with_embedding_model(self, model: str) -> set[str]:
        """Return raw artifact ids already embedded with a model."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT artifact_id FROM raw_artifact_embeddings WHERE lower(model) = lower(?)",
                (model,),
            ).fetchall()
        return {row["artifact_id"] for row in rows}

    def clear_all_embeddings(self) -> tuple[int, int]:
        """Delete all memory and raw artifact embeddings."""
        with self._connect() as connection:
            memory_count = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_embeddings"
            ).fetchone()["count"]
            raw_count = connection.execute(
                "SELECT COUNT(*) AS count FROM raw_artifact_embeddings"
            ).fetchone()["count"]
            connection.execute("DELETE FROM memory_embeddings")
            connection.execute("DELETE FROM raw_artifact_embeddings")
        return int(memory_count), int(raw_count)

    def get_raw_artifact_embedding(self, artifact_id: str) -> Embedding | None:
        """Return one stored raw artifact embedding by artifact id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM raw_artifact_embeddings WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return Embedding(
            provider=row["provider"],
            model=row["model"],
            vector=[float(value) for value in loads(row["vector_json"])],
        )

    def semantic_search_raw_artifacts(
        self,
        query_embedding: Embedding,
        limit: int,
    ) -> list[tuple[RawArtifact, float]]:
        """Search raw artifact embeddings by exact cosine similarity."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, e.vector_json
                FROM raw_artifact_embeddings e
                JOIN raw_artifacts r ON r.id = e.artifact_id
                """
            ).fetchall()
        scored = [
            (
                self._raw_artifact_from_row(row),
                self._cosine_similarity(query_embedding.vector, self._vector_from_row(row)),
            )
            for row in rows
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    def get_raw_artifact(self, artifact_id: str) -> RawArtifact | None:
        """Return one raw artifact by id, if it exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM raw_artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return self._raw_artifact_from_row(row)

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

    @staticmethod
    def _recreate_raw_schema_if_legacy(connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(raw_artifacts)").fetchall()
        columns = {row["name"] for row in rows}
        expected = {
            "id",
            "provider",
            "source_path",
            "source_conversation_id",
            "created_at",
            "updated_at",
            "title",
            "workspace",
            "content",
        }
        if expected.issubset(columns):
            return
        connection.executescript(
            """
            DROP TABLE IF EXISTS raw_artifact_fts;
            DROP TABLE IF EXISTS raw_artifact_span_fts;
            DROP TABLE IF EXISTS raw_artifact_spans;
            DROP TABLE IF EXISTS raw_artifact_embeddings;
            DROP TABLE IF EXISTS raw_artifacts;
            """
        )

    @staticmethod
    def _ensure_raw_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_artifacts (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_conversation_id TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                title TEXT,
                workspace TEXT,
                content TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_artifact_spans (
                id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                span_index INTEGER NOT NULL,
                message_index INTEGER NOT NULL,
                message_id TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (artifact_id) REFERENCES raw_artifacts(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS raw_artifact_span_fts
            USING fts5(span_id UNINDEXED, content);

            CREATE TABLE IF NOT EXISTS raw_artifact_embeddings (
                artifact_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (artifact_id) REFERENCES raw_artifacts(id) ON DELETE CASCADE
            );
            """
        )

    def _delete_stale_raw_artifacts(
        self,
        connection: sqlite3.Connection,
        artifacts: list[RawArtifact],
        providers: list[str] | None,
    ) -> None:
        new_ids = {artifact.id for artifact in artifacts}
        stale_ids = [
            artifact_id
            for artifact_id in self._raw_artifact_ids_for_providers(connection, providers)
            if artifact_id not in new_ids
        ]
        if not stale_ids:
            return
        connection.executemany(
            "DELETE FROM raw_artifact_span_fts WHERE span_id = ?",
            [(span_id,) for span_id in self._raw_span_ids_for_artifact_ids(connection, stale_ids)],
        )
        connection.executemany(
            "DELETE FROM raw_artifact_spans WHERE artifact_id = ?",
            [(artifact_id,) for artifact_id in stale_ids],
        )
        connection.executemany(
            "DELETE FROM raw_artifact_embeddings WHERE artifact_id = ?",
            [(artifact_id,) for artifact_id in stale_ids],
        )
        connection.executemany(
            "DELETE FROM raw_artifacts WHERE id = ?",
            [(artifact_id,) for artifact_id in stale_ids],
        )

    @staticmethod
    def _rebuild_raw_artifact_span_fts(connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM raw_artifact_span_fts")
        connection.execute(
            """
            INSERT INTO raw_artifact_span_fts (span_id, content)
            SELECT id, content FROM raw_artifact_spans
            """
        )

    @staticmethod
    def _raw_span_ids_for_artifact_ids(
        connection: sqlite3.Connection,
        artifact_ids: list[str],
    ) -> list[str]:
        span_ids: list[str] = []
        for artifact_id in artifact_ids:
            rows = connection.execute(
                "SELECT id FROM raw_artifact_spans WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchall()
            span_ids.extend(row["id"] for row in rows)
        return span_ids

    @staticmethod
    def _raw_artifact_ids_for_providers(
        connection: sqlite3.Connection,
        providers: list[str] | None,
    ) -> list[str]:
        if providers is None:
            rows = connection.execute("SELECT id FROM raw_artifacts").fetchall()
            return [row["id"] for row in rows]
        artifact_ids: list[str] = []
        for provider in providers:
            rows = connection.execute(
                "SELECT id FROM raw_artifacts WHERE provider = ?",
                (provider,),
            ).fetchall()
            artifact_ids.extend(row["id"] for row in rows)
        return artifact_ids

    @staticmethod
    def _missing_raw_artifact_ids(
        connection: sqlite3.Connection,
        embeddings: list[tuple[str, Embedding]],
    ) -> list[str]:
        missing: list[str] = []
        for artifact_id, _embedding in embeddings:
            row = connection.execute(
                "SELECT 1 FROM raw_artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                missing.append(artifact_id)
        return missing

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
    def _validate_raw_artifact(artifact: RawArtifact) -> None:
        if artifact.content.strip():
            return
        msg = "Raw artifact content cannot be empty"
        raise ValueError(msg)

    @staticmethod
    def _validate_raw_artifact_span(span: RawArtifactSpan) -> None:
        if not span.content.strip():
            msg = "Raw artifact span content cannot be empty"
            raise ValueError(msg)
        if span.start_offset < 0:
            msg = "Raw artifact span start_offset cannot be negative"
            raise ValueError(msg)
        if span.end_offset <= span.start_offset:
            msg = "Raw artifact span end_offset must be greater than start_offset"
            raise ValueError(msg)

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
    def _raw_artifact_values(
        artifact: RawArtifact,
    ) -> tuple[str, str, str, str, str | None, str | None, str | None, str | None, str]:
        return (
            artifact.id,
            artifact.provider,
            artifact.source_path,
            artifact.source_conversation_id,
            artifact.created_at,
            artifact.updated_at,
            artifact.title,
            artifact.workspace,
            artifact.content.strip(),
        )

    @staticmethod
    def _raw_artifact_span_values(
        span: RawArtifactSpan,
    ) -> tuple[str, str, int, int, str, str, str | None, int, int, str]:
        return (
            span.id,
            span.artifact_id,
            span.span_index,
            span.message_index,
            span.message_id,
            span.role,
            span.created_at,
            span.start_offset,
            span.end_offset,
            span.content.strip(),
        )

    @staticmethod
    def _raw_artifact_from_row(row: sqlite3.Row) -> RawArtifact:
        row_data: dict[str, Any] = dict(row)
        return RawArtifact(
            id=row_data["id"],
            provider=row_data["provider"],
            source_path=row_data["source_path"],
            source_conversation_id=row_data["source_conversation_id"],
            created_at=row_data["created_at"],
            updated_at=row_data["updated_at"],
            title=row_data["title"],
            workspace=row_data["workspace"],
            content=row_data["content"],
        )

    @staticmethod
    def _raw_artifact_span_from_row(row: sqlite3.Row) -> RawArtifactSpan:
        row_data: dict[str, Any] = dict(row)
        return RawArtifactSpan(
            id=row_data["span_id"],
            artifact_id=row_data["artifact_id"],
            span_index=row_data["span_index"],
            message_index=row_data["message_index"],
            message_id=row_data["message_id"],
            role=row_data["role"],
            created_at=row_data["span_created_at"],
            start_offset=row_data["start_offset"],
            end_offset=row_data["end_offset"],
            content=row_data["span_content"],
        )

    def _raw_artifact_search_match_from_row(self, row: sqlite3.Row) -> RawArtifactSearchMatch:
        return RawArtifactSearchMatch(
            artifact=self._raw_artifact_from_row(row),
            span=self._raw_artifact_span_from_row(row),
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = [term.strip('"') for term in query.split() if term.strip('"')]
        return " OR ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _vector_from_row(row: sqlite3.Row) -> list[float]:
        vector = loads(row["vector_json"])
        if not isinstance(vector, list):
            msg = "Stored embedding vector was not a JSON array"
            raise TypeError(msg)
        return [float(value) for value in vector]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return 0.0
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        dot_product = sum(
            left_value * right_value for left_value, right_value in zip(left, right, strict=True)
        )
        return dot_product / (left_norm * right_norm)
