from pathlib import Path

from memories.embedder import Embedding
from memories.models import AddMemory
from memories.service import MemoryService
from memories.sqlite_store import SQLiteMemoryStore


class RecordingEmbedder:
    def __init__(self) -> None:
        """Create an embedder that records inputs."""
        self.inputs: list[str] = []

    def embed(self, text: str) -> Embedding:
        self.inputs.append(text)
        return Embedding(provider="fake", model="tiny", vector=[0.1, 0.2])


def test_service_stores_embedding_when_embedder_is_configured(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    embedder = RecordingEmbedder()
    service = MemoryService(store, embedder=embedder)

    memory = service.add_memory(
        AddMemory(
            content="Aaryan wants embedding providers to be swappable.",
            source="codex",
            memory_type="decision",
        )
    )

    stored_embedding = store.get_embedding(memory.id)

    assert embedder.inputs == ["Aaryan wants embedding providers to be swappable."]
    assert stored_embedding == Embedding(provider="fake", model="tiny", vector=[0.1, 0.2])
