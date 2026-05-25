from pathlib import Path

from memories.embedder import Embedding, OpenRouterEmbedder
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
            content="The operator wants embedding providers to be swappable.",
            source="codex",
            memory_type="decision",
        )
    )

    stored_embedding = store.get_embedding(memory.id)

    assert embedder.inputs == ["The operator wants embedding providers to be swappable."]
    assert stored_embedding == Embedding(provider="fake", model="tiny", vector=[0.1, 0.2])


def test_openrouter_embedder_can_embed_texts_in_one_batch() -> None:
    def post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        del url, headers, timeout
        assert payload["input"] == ["first text", "second text"]
        return {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ],
            "model": "fake/embedding",
        }

    embedder = OpenRouterEmbedder(api_key="secret", model="fake/embedding", post_json=post_json)

    embeddings = embedder.embed_many(["first text", "second text"])

    assert embeddings == [
        Embedding(provider="openrouter", model="fake/embedding", vector=[0.1, 0.2]),
        Embedding(provider="openrouter", model="fake/embedding", vector=[0.3, 0.4]),
    ]
