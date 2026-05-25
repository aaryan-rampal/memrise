from pathlib import Path

from memories.embedder import Embedding
from memories.models import AddMemory, RawArtifact
from memories.service import MemoryService
from memories.sqlite_store import SQLiteMemoryStore


def make_service(db_path: Path) -> MemoryService:
    store = SQLiteMemoryStore(db_path)
    store.initialize()
    return MemoryService(store)


def test_add_memory_persists_validated_record_with_unix_timestamps(tmp_path: Path) -> None:
    service = make_service(tmp_path / "memories.sqlite3")

    memory = service.add_memory(
        AddMemory(
            content="The operator prefers small dogfoodable loops for local infrastructure.",
            source="codex",
            memory_type="preference",
            importance=0.8,
        )
    )

    assert memory.id
    assert (
        memory.content == "The operator prefers small dogfoodable loops for local infrastructure."
    )
    assert memory.source.value == "codex"
    assert memory.memory_type.value == "preference"
    assert memory.importance == 0.8
    assert isinstance(memory.created_at, int)
    assert memory.created_at == memory.updated_at


def test_recent_returns_newest_memories_first(tmp_path: Path) -> None:
    service = make_service(tmp_path / "memories.sqlite3")
    first = service.add_memory(
        AddMemory(content="First memory", source="codex", memory_type="fact", importance=0.5)
    )
    second = service.add_memory(
        AddMemory(content="Second memory", source="codex", memory_type="fact", importance=0.5)
    )

    recent = service.recent(limit=2)

    assert [memory.id for memory in recent] == [second.id, first.id]


def test_search_uses_sqlite_fts_content_matching(tmp_path: Path) -> None:
    service = make_service(tmp_path / "memories.sqlite3")
    service.add_memory(
        AddMemory(
            content="The operator wants the memory daemon to start with SQLite and FTS5.",
            source="codex",
            memory_type="decision",
            importance=0.9,
        )
    )
    service.add_memory(
        AddMemory(
            content="The operator uses a separate project for playlist automation.",
            source="codex",
            memory_type="fact",
            importance=0.3,
        )
    )

    results = service.search("SQLite FTS5", limit=5)

    assert [memory.memory_type.value for memory in results] == ["decision"]
    assert "memory daemon" in results[0].content


def test_update_changes_content_and_updated_timestamp(tmp_path: Path) -> None:
    service = make_service(tmp_path / "memories.sqlite3")
    memory = service.add_memory(
        AddMemory(content="Original memory", source="codex", memory_type="fact", importance=0.5)
    )

    updated = service.update_memory(memory.id, content="Updated memory", importance=0.7)

    assert updated.content == "Updated memory"
    assert updated.importance == 0.7
    assert updated.updated_at >= memory.updated_at


def test_delete_removes_memory_from_recent_and_search(tmp_path: Path) -> None:
    service = make_service(tmp_path / "memories.sqlite3")
    memory = service.add_memory(
        AddMemory(content="Delete this SQLite memory", source="codex", memory_type="fact")
    )

    assert service.delete_memory(memory.id) is True

    assert service.recent(limit=5) == []
    assert service.search("SQLite", limit=5) == []


def test_semantic_search_ranks_memories_by_cosine_similarity(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    first = store.add(AddMemory(content="Memory about local recovery"))
    second = store.add(AddMemory(content="Memory about Spotify automation"))
    store.save_embedding(first.id, Embedding(provider="fake", model="tiny", vector=[1.0, 0.0]))
    store.save_embedding(second.id, Embedding(provider="fake", model="tiny", vector=[0.0, 1.0]))

    results = store.semantic_search(Embedding(provider="fake", model="tiny", vector=[0.8, 0.2]), 2)

    assert [memory.id for memory, score in results] == [first.id, second.id]
    assert results[0][1] > results[1][1]


def test_raw_artifact_search_uses_separate_fts_index(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    store.add(AddMemory(content="Memory about durable preference storage"))
    store.upsert_raw_artifact(
        RawArtifact(
            id="raw-1",
            provider="codex",
            source_path="data/canonical/chats/codex.jsonl",
            source_conversation_id="session-1",
            message_id="message-1",
            role="user",
            created_at="2026-05-24T00:00:00Z",
            content="Raw transcript mentions canonical artifacts.",
        )
    )

    raw_results = store.search_raw_artifacts("canonical artifacts", limit=5)
    memory_results = store.search("canonical artifacts", limit=5)

    assert [artifact.id for artifact in raw_results] == ["raw-1"]
    assert memory_results == []


def test_raw_artifact_semantic_search_ranks_by_embedding(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    first = RawArtifact(
        id="raw-1",
        provider="codex",
        source_path="data/canonical/chats/codex.jsonl",
        source_conversation_id="session-1",
        message_id="message-1",
        role="user",
        created_at=None,
        content="Raw chat about git recovery",
    )
    second = RawArtifact(
        id="raw-2",
        provider="claude-code",
        source_path="data/canonical/chats/claude-code.jsonl",
        source_conversation_id="session-2",
        message_id="message-2",
        role="assistant",
        created_at=None,
        content="Raw chat about playlist automation",
    )
    store.upsert_raw_artifact(first)
    store.upsert_raw_artifact(second)
    store.save_raw_artifact_embedding(
        first.id,
        Embedding(provider="fake", model="tiny", vector=[1.0, 0.0]),
    )
    store.save_raw_artifact_embedding(
        second.id,
        Embedding(provider="fake", model="tiny", vector=[0.0, 1.0]),
    )

    results = store.semantic_search_raw_artifacts(
        Embedding(provider="fake", model="tiny", vector=[0.9, 0.1]),
        2,
    )

    assert [artifact.id for artifact, score in results] == ["raw-1", "raw-2"]
    assert results[0][1] > results[1][1]


def test_raw_artifact_embedding_model_lookup_is_case_insensitive(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    artifact = RawArtifact(
        id="raw-1",
        provider="codex",
        source_path="data/canonical/chats/codex.jsonl",
        source_conversation_id="session-1",
        message_id="message-1",
        role="user",
        created_at=None,
        content="Raw chat content",
    )
    store.upsert_raw_artifact(artifact)
    store.save_raw_artifact_embedding(
        artifact.id,
        Embedding(provider="openrouter", model="Qwen/Qwen3-Embedding-8B", vector=[1.0]),
    )

    assert store.raw_artifact_ids_with_embedding_model("qwen/qwen3-embedding-8b") == {"raw-1"}
