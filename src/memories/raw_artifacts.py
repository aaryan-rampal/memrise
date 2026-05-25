from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from memories.embedder import Embedder, Embedding
from memories.models import RawArtifact
from memories.sqlite_store import SQLiteMemoryStore

JsonObject = dict[str, Any]
EMBEDDING_BATCH_SIZE = 128
ProgressLog = Callable[[str], None]


@dataclass(frozen=True)
class RawArtifactIndexOptions:
    """Options for indexing canonical chat artifacts."""

    providers: list[str] | None = None
    embedder: Embedder | None = None
    reset: bool = True
    rebuild_embeddings: bool = False
    log: ProgressLog | None = None


def index_canonical_chats(
    store: SQLiteMemoryStore,
    canonical_dir: Path | str = Path("data/canonical/chats"),
    options: RawArtifactIndexOptions | None = None,
) -> int:
    """Index canonical chat JSONL files as searchable raw artifacts."""
    options = options or RawArtifactIndexOptions()
    artifacts = _deduplicate_artifacts(
        [
            artifact
            for path in _canonical_paths(Path(canonical_dir), options.providers)
            for conversation in _jsonl_objects(path)
            for artifact in _conversation_artifacts(conversation)
        ]
    )
    if options.reset:
        store.replace_raw_artifacts(artifacts, options.providers)
    else:
        store.upsert_raw_artifacts(artifacts)
    _log(options.log, f"raw-artifacts\t{len(artifacts)}")
    if options.embedder is None:
        return len(artifacts)
    if options.rebuild_embeddings:
        store.clear_raw_artifact_embeddings(options.providers)
    pending = _pending_embedding_artifacts(store, artifacts, options.embedder)
    skipped = len(artifacts) - len(pending)
    _log(options.log, f"raw-embedding-skip\t{skipped}")
    batches = _batches(pending)
    total_batches = len(batches)
    embedded = 0
    for batch_number, batch in enumerate(batches, start=1):
        _log(options.log, f"raw-embedding-batch\t{batch_number}\t{total_batches}\t{len(batch)}")
        embeddings = _embed_many(options.embedder, [artifact.content for artifact in batch])
        store.save_raw_artifact_embeddings(
            [
                (artifact.id, embedding)
                for artifact, embedding in zip(batch, embeddings, strict=True)
            ]
        )
        embedded += len(batch)
        _log(options.log, f"raw-embedding-progress\t{embedded}\t{len(pending)}")
    _log(options.log, f"raw-embedding-complete\t{len(artifacts)}\t{skipped}\t{embedded}")
    return len(artifacts)


def _pending_embedding_artifacts(
    store: SQLiteMemoryStore,
    artifacts: list[RawArtifact],
    embedder: Embedder,
) -> list[RawArtifact]:
    model = _embedder_model(embedder)
    if model is None:
        return artifacts
    embedded_ids = store.raw_artifact_ids_with_embedding_model(model)
    return [artifact for artifact in artifacts if artifact.id not in embedded_ids]


def _embedder_model(embedder: Embedder) -> str | None:
    model = getattr(embedder, "model", None)
    if isinstance(model, str):
        return model
    private_model = getattr(embedder, "_model", None)
    if isinstance(private_model, str):
        return private_model
    return None


def _log(log: ProgressLog | None, message: str) -> None:
    if log is not None:
        log(message)


def _embed_many(embedder: Embedder, texts: list[str]) -> list[Embedding]:
    batch_method = getattr(embedder, "embed_many", None)
    if callable(batch_method):
        embeddings = batch_method(texts)
        if not isinstance(embeddings, list):
            msg = "Batch embedder returned a non-list result"
            raise TypeError(msg)
        return embeddings
    return [embedder.embed(text) for text in texts]


def _deduplicate_artifacts(artifacts: list[RawArtifact]) -> list[RawArtifact]:
    by_id: dict[str, RawArtifact] = {}
    for artifact in artifacts:
        by_id[artifact.id] = artifact
    return list(by_id.values())


def _batches(artifacts: list[RawArtifact]) -> list[list[RawArtifact]]:
    return [
        artifacts[index : index + EMBEDDING_BATCH_SIZE]
        for index in range(0, len(artifacts), EMBEDDING_BATCH_SIZE)
    ]


def _canonical_paths(canonical_dir: Path, providers: list[str] | None) -> list[Path]:
    if providers is None:
        return sorted(canonical_dir.glob("*.jsonl"))
    return [canonical_dir / f"{provider}.jsonl" for provider in providers]


def _conversation_artifacts(conversation: JsonObject) -> list[RawArtifact]:
    provider = str(conversation.get("provider", "unknown"))
    source_path = str(conversation.get("source_path", ""))
    source_conversation_id = str(conversation.get("source_conversation_id", ""))
    artifacts: list[RawArtifact] = []
    for message in _list_of_objects(conversation.get("messages")):
        content = _artifact_content(message)
        if not content:
            continue
        message_id = str(message.get("id", ""))
        artifacts.append(
            RawArtifact(
                id=_artifact_id(provider, source_conversation_id, message_id, content),
                provider=provider,
                source_path=source_path,
                source_conversation_id=source_conversation_id,
                message_id=message_id,
                role=str(message.get("role", "unknown")),
                created_at=_optional_str(message.get("created_at")),
                content=content,
            )
        )
    return artifacts


def _artifact_content(message: JsonObject) -> str:
    parts = [
        rendered
        for item in _list_of_objects(message.get("content"))
        if (rendered := _render_content_item(item))
    ]
    return "\n".join(parts).strip()


def _render_content_item(item: JsonObject) -> str:
    renderers = {
        "text": _render_text,
        "tool_call": _render_tool_call,
        "tool_result": _render_tool_result,
        "attachment": _render_attachment,
        "file": _render_file,
        "reasoning": _render_reasoning,
    }
    renderer = renderers.get(str(item.get("type", "")))
    if renderer is None:
        return ""
    return renderer(item)


def _render_text(item: JsonObject) -> str:
    return str(item.get("text", "")).strip()


def _render_tool_call(item: JsonObject) -> str:
    name = _optional_str(item.get("name")) or "unknown"
    return f"tool call: {name} called"


def _render_tool_result(item: JsonObject) -> str:
    del item
    return "tool result: completed"


def _render_attachment(item: JsonObject) -> str:
    name = _optional_str(item.get("name")) or _optional_str(item.get("id")) or "unknown"
    return f"attachment: {name}"


def _render_file(item: JsonObject) -> str:
    name = _optional_str(item.get("name")) or _optional_str(item.get("id")) or "unknown"
    return f"file: {name}"


def _render_reasoning(item: JsonObject) -> str:
    del item
    return "reasoning"


def _artifact_id(
    provider: str,
    source_conversation_id: str,
    message_id: str,
    content: str,
) -> str:
    stable_key = f"{provider}\0{source_conversation_id}\0{message_id}\0{content}"
    return hashlib.sha256(stable_key.encode("utf-8")).hexdigest()


def _jsonl_objects(path: Path) -> list[JsonObject]:
    if not path.exists():
        return []
    records: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            msg = f"Expected JSON object in {path}:{line_number}"
            raise TypeError(msg)
        records.append(data)
    return records


def _list_of_objects(value: object) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [cast("JsonObject", item) for item in value if isinstance(item, dict)]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
