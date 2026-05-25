import json
from pathlib import Path

from memories.embedder import Embedding
from memories.models import RawArtifact
from memories.raw_artifacts import RawArtifactIndexOptions, index_canonical_chats
from memories.sqlite_store import SQLiteMemoryStore


class RecordingEmbedder:
    def __init__(self) -> None:
        """Create an embedder that records inputs."""
        self.inputs: list[str] = []

    def embed(self, text: str) -> Embedding:
        self.inputs.append(text)
        return Embedding(provider="fake", model="tiny", vector=[0.1, 0.2])


class BatchRecordingEmbedder:
    model = "fake/batch"

    def __init__(self) -> None:
        """Create a batch embedder that records batch inputs."""
        self.batches: list[list[str]] = []

    def embed(self, text: str) -> Embedding:
        msg = f"single embed should not be called for {text}"
        raise AssertionError(msg)

    def embed_many(self, texts: list[str]) -> list[Embedding]:
        self.batches.append(texts)
        return [
            Embedding(provider="fake", model=self.model, vector=[float(index)])
            for index, _text in enumerate(texts)
        ]


def test_index_canonical_chats_stores_sanitized_raw_artifacts(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "codex.jsonl").write_text(
        json.dumps(
            {
                "provider": "codex",
                "source_path": "data/raw/chats/codex/rollout.jsonl",
                "source_conversation_id": "session-1",
                "messages": [
                    {
                        "id": "call-1",
                        "role": "tool",
                        "created_at": "2026-05-24T00:00:00Z",
                        "content": [
                            {
                                "type": "tool_call",
                                "name": "exec_command",
                                "status": "called",
                            }
                        ],
                    }
                ],
            }
        )
        + "\n"
    )
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()

    count = index_canonical_chats(
        store,
        canonical_dir,
        RawArtifactIndexOptions(providers=["codex"]),
    )

    results = store.search_raw_artifacts("exec_command", limit=5)
    assert count == 1
    assert len(results) == 1
    assert results[0].provider == "codex"
    assert results[0].content == "tool call: exec_command called"


def test_index_canonical_chats_embeds_each_raw_artifact(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "claude-code.jsonl").write_text(
        json.dumps(
            {
                "provider": "claude-code",
                "source_path": "data/raw/chats/claude-code/session.jsonl",
                "source_conversation_id": "session-1",
                "messages": [
                    {
                        "id": "message-1",
                        "role": "user",
                        "created_at": None,
                        "content": [{"type": "text", "text": "raw text to embed"}],
                    }
                ],
            }
        )
        + "\n"
    )
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    embedder = RecordingEmbedder()

    count = index_canonical_chats(
        store,
        canonical_dir,
        RawArtifactIndexOptions(providers=["claude-code"], embedder=embedder),
    )
    [artifact] = store.search_raw_artifacts("embed", limit=5)

    assert count == 1
    assert embedder.inputs == ["raw text to embed"]
    assert store.get_raw_artifact_embedding(artifact.id) == Embedding(
        provider="fake",
        model="tiny",
        vector=[0.1, 0.2],
    )


def test_index_canonical_chats_replaces_partial_raw_artifact_index(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "codex.jsonl").write_text(
        json.dumps(
            {
                "provider": "codex",
                "source_path": "data/raw/chats/codex/new.jsonl",
                "source_conversation_id": "session-new",
                "messages": [
                    {
                        "id": "message-new",
                        "role": "user",
                        "created_at": None,
                        "content": [{"type": "text", "text": "new canonical content"}],
                    }
                ],
            }
        )
        + "\n"
    )
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    store.upsert_raw_artifact(
        RawArtifact(
            id="stale",
            provider="codex",
            source_path="old",
            source_conversation_id="old",
            message_id="old",
            role="user",
            created_at=None,
            content="stale interrupted content",
        )
    )

    count = index_canonical_chats(
        store,
        canonical_dir,
        RawArtifactIndexOptions(providers=["codex"], reset=True),
    )

    assert count == 1
    assert store.search_raw_artifacts("stale", limit=5) == []
    assert len(store.search_raw_artifacts("canonical", limit=5)) == 1


def test_index_canonical_chats_deduplicates_repeated_canonical_messages(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    duplicate_message = {
        "id": "message-1",
        "role": "user",
        "created_at": None,
        "content": [{"type": "text", "text": "duplicated canonical content"}],
    }
    (canonical_dir / "codex.jsonl").write_text(
        json.dumps(
            {
                "provider": "codex",
                "source_path": "data/raw/chats/codex/new.jsonl",
                "source_conversation_id": "session-1",
                "messages": [duplicate_message, duplicate_message],
            }
        )
        + "\n"
    )
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()

    count = index_canonical_chats(
        store,
        canonical_dir,
        RawArtifactIndexOptions(providers=["codex"]),
    )

    assert count == 1
    assert len(store.search_raw_artifacts("duplicated", limit=5)) == 1


def test_index_canonical_chats_skips_existing_embeddings_and_logs_progress(
    tmp_path: Path,
) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    messages = [
        {
            "id": "message-1",
            "role": "user",
            "created_at": None,
            "content": [{"type": "text", "text": "already embedded"}],
        },
        {
            "id": "message-2",
            "role": "user",
            "created_at": None,
            "content": [{"type": "text", "text": "needs embedding"}],
        },
    ]
    (canonical_dir / "codex.jsonl").write_text(
        json.dumps(
            {
                "provider": "codex",
                "source_path": "data/raw/chats/codex/new.jsonl",
                "source_conversation_id": "session-1",
                "messages": messages,
            }
        )
        + "\n"
    )
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()
    index_canonical_chats(store, canonical_dir, RawArtifactIndexOptions(providers=["codex"]))
    [existing] = store.search_raw_artifacts("already", limit=5)
    store.save_raw_artifact_embedding(
        existing.id,
        Embedding(provider="fake", model="fake/batch", vector=[9.0]),
    )
    embedder = BatchRecordingEmbedder()
    logs: list[str] = []

    count = index_canonical_chats(
        store,
        canonical_dir,
        RawArtifactIndexOptions(
            providers=["codex"],
            embedder=embedder,
            log=logs.append,
        ),
    )

    assert count == 2
    assert embedder.batches == [["needs embedding"]]
    assert store.raw_artifact_embedding_count() == 2
    assert any("raw-embedding-skip\t1" in line for line in logs)
    assert any("raw-embedding-batch\t1\t1\t1" in line for line in logs)
    assert any("raw-embedding-complete\t2\t1\t1" in line for line in logs)
