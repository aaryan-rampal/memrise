from pathlib import Path

from memories.models import AddMemory
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
            content="Aaryan prefers small dogfoodable loops for personal infrastructure.",
            source="codex",
            memory_type="preference",
            importance=0.8,
        )
    )

    assert memory.id
    assert memory.content == "Aaryan prefers small dogfoodable loops for personal infrastructure."
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
            content="Aaryan wants the memory daemon to start with SQLite and FTS5.",
            source="codex",
            memory_type="decision",
            importance=0.9,
        )
    )
    service.add_memory(
        AddMemory(
            content="Aaryan uses a separate project for Spotify automation.",
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
