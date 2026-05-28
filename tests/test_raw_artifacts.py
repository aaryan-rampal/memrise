import json
from pathlib import Path

from memories.embedder import Embedding
from memories.models import RawArtifact, RawArtifactSpan
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


def make_raw_artifact(content: str, artifact_id: str = "raw-1") -> RawArtifact:
    return RawArtifact(
        id=artifact_id,
        provider="codex",
        source_path="data/canonical/chats/codex.jsonl",
        source_conversation_id=artifact_id,
        created_at=None,
        updated_at=None,
        title=None,
        workspace=None,
        content=content,
    )


def make_raw_span(content: str, artifact_id: str = "raw-1") -> RawArtifactSpan:
    return RawArtifactSpan(
        id=f"{artifact_id}-span-1",
        artifact_id=artifact_id,
        span_index=0,
        message_index=0,
        message_id="message-1",
        role="user",
        created_at=None,
        start_offset=0,
        end_offset=len(content),
        content=content,
    )


def test_index_canonical_chats_stores_conversations_with_searchable_spans(
    tmp_path: Path,
) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "codex.jsonl").write_text(
        json.dumps(
            {
                "provider": "codex",
                "source_path": "data/raw/chats/codex/rollout.jsonl",
                "source_conversation_id": "session-1",
                "title": "Searchable session",
                "created_at": "2026-05-24T00:00:00Z",
                "updated_at": "2026-05-24T00:10:00Z",
                "workspace": "/workspace/project",
                "messages": [
                    {
                        "id": "message-1",
                        "role": "user",
                        "created_at": "2026-05-24T00:01:00Z",
                        "content": [{"type": "text", "text": "first request"}],
                    },
                    {
                        "id": "message-2",
                        "role": "assistant",
                        "created_at": "2026-05-24T00:02:00Z",
                        "content": [
                            {
                                "type": "tool_call",
                                "name": "exec_command",
                                "status": "called",
                            }
                        ],
                    },
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
    assert store.raw_artifact_count() == 1
    assert results[0].artifact.provider == "codex"
    assert results[0].artifact.source_conversation_id == "session-1"
    assert results[0].artifact.content == (
        "user: first request\n\nassistant: tool call: exec_command called"
    )
    assert results[0].span.message_id == "message-2"
    assert results[0].span.role == "assistant"
    assert results[0].span.start_offset == 21
    assert results[0].span.content == "assistant: tool call: exec_command called"


def test_index_canonical_chats_splits_large_messages_into_searchable_spans(
    tmp_path: Path,
) -> None:
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    long_text = "early context " + ("filler " * 900) + "late-needle final context"
    (canonical_dir / "codex.jsonl").write_text(
        json.dumps(
            {
                "provider": "codex",
                "source_path": "data/raw/chats/codex/rollout.jsonl",
                "source_conversation_id": "session-1",
                "messages": [
                    {
                        "id": "message-1",
                        "role": "user",
                        "created_at": "2026-05-24T00:01:00Z",
                        "content": [{"type": "text", "text": long_text}],
                    }
                ],
            }
        )
        + "\n"
    )
    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    store.initialize()

    index_canonical_chats(store, canonical_dir, RawArtifactIndexOptions(providers=["codex"]))

    [match] = store.search_raw_artifacts("late-needle", limit=5)
    assert match.artifact.content == f"user: {long_text}"
    assert 0 < match.span.start_offset < match.span.end_offset
    assert match.span.end_offset <= len(match.artifact.content)
    assert len(match.span.content) < len(match.artifact.content)
    assert "late-needle" in match.span.content


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
    [match] = store.search_raw_artifacts("embed", limit=5)

    assert count == 1
    assert embedder.inputs == ["user: raw text to embed"]
    assert store.get_raw_artifact_embedding(match.artifact.id) == Embedding(
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
        make_raw_artifact("stale interrupted content", artifact_id="stale"),
        [make_raw_span("stale interrupted content", artifact_id="stale")],
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
        existing.artifact.id,
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

    assert count == 1
    assert embedder.batches == []
    assert store.raw_artifact_embedding_count() == 1
    assert any("raw-embedding-skip\t1" in line for line in logs)
    assert any("raw-embedding-complete\t1\t1\t0" in line for line in logs)
