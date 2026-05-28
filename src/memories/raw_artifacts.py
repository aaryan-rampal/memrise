from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from memories.embedder import Embedder, Embedding
from memories.models import RawArtifact, RawArtifactSpan
from memories.sqlite_store import SQLiteMemoryStore

JsonObject = dict[str, Any]
EMBEDDING_BATCH_SIZE = 128
RAW_SPAN_MAX_CHARS = 4000
RAW_SPAN_OVERLAP_CHARS = 200
ProgressLog = Callable[[str], None]


@dataclass(frozen=True)
class IndexedRawArtifact:
    """Raw artifact with its searchable spans."""

    artifact: RawArtifact
    spans: list[RawArtifactSpan]


@dataclass(frozen=True)
class SpanSource:
    """Message text prepared for span splitting."""

    artifact_id: str
    message_index: int
    message_id: str
    role: str
    created_at: str | None
    start_offset: int
    text: str


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
    canonical_path = Path(canonical_dir)
    paths = _canonical_paths(canonical_path, options.providers)
    logger.info(
        "raw_artifact_index_start canonical_dir={} providers={} files={}",
        canonical_path,
        options.providers or "all",
        len(paths),
    )
    indexed = _deduplicate_artifacts(_artifacts_from_paths(paths))
    artifacts = [item.artifact for item in indexed]
    spans = [span for item in indexed for span in item.spans]
    logger.info("raw_artifact_index_loaded artifacts={}", len(artifacts))
    if options.reset:
        logger.info("raw_artifact_index_replace_start providers={}", options.providers or "all")
        store.replace_raw_artifacts(artifacts, spans, options.providers)
        logger.info("raw_artifact_index_replace_complete artifacts={}", len(artifacts))
    else:
        logger.info("raw_artifact_index_upsert_start artifacts={}", len(artifacts))
        store.upsert_raw_artifacts(artifacts, spans)
        logger.info("raw_artifact_index_upsert_complete artifacts={}", len(artifacts))
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


def _artifacts_from_paths(paths: list[Path]) -> list[IndexedRawArtifact]:
    artifacts: list[IndexedRawArtifact] = []
    for path in paths:
        logger.info("raw_artifact_file_start path={}", path)
        conversations = _jsonl_objects(path)
        file_artifacts = [
            artifact
            for conversation in conversations
            if (artifact := _conversation_artifact(conversation)) is not None
        ]
        artifacts.extend(file_artifacts)
        logger.info(
            "raw_artifact_file_complete path={} conversations={} artifacts={}",
            path,
            len(conversations),
            len(file_artifacts),
        )
    return artifacts


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


def _deduplicate_artifacts(artifacts: list[IndexedRawArtifact]) -> list[IndexedRawArtifact]:
    by_id: dict[str, IndexedRawArtifact] = {}
    for indexed in artifacts:
        by_id[indexed.artifact.id] = indexed
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


def _conversation_artifact(conversation: JsonObject) -> IndexedRawArtifact | None:
    provider = str(conversation.get("provider", "unknown"))
    source_path = str(conversation.get("source_path", ""))
    source_conversation_id = str(conversation.get("source_conversation_id", ""))
    artifact_id = _artifact_id(provider, source_conversation_id, source_path)
    content_parts: list[str] = []
    spans: list[RawArtifactSpan] = []
    seen_messages: set[tuple[str, str]] = set()
    offset = 0
    span_index = 0
    for message_index, message in enumerate(_list_of_objects(conversation.get("messages"))):
        content = _artifact_content(message)
        if not content:
            continue
        message_id = str(message.get("id", ""))
        message_key = (message_id, content)
        if message_key in seen_messages:
            continue
        seen_messages.add(message_key)
        role = str(message.get("role", "unknown"))
        message_text = f"{role}: {content}"
        if content_parts:
            offset += 2
        content_parts.append(message_text)
        start_offset = offset
        end_offset = start_offset + len(message_text)
        message_spans = _message_spans(
            SpanSource(
                artifact_id=artifact_id,
                message_index=message_index,
                message_id=message_id,
                role=role,
                created_at=_optional_str(message.get("created_at")),
                start_offset=start_offset,
                text=message_text,
            ),
            span_index,
        )
        spans.extend(message_spans)
        span_index += len(message_spans)
        offset = end_offset
    full_content = "\n\n".join(content_parts)
    if not full_content:
        return None
    return IndexedRawArtifact(
        artifact=RawArtifact(
            id=artifact_id,
            provider=provider,
            source_path=source_path,
            source_conversation_id=source_conversation_id,
            created_at=_optional_str(conversation.get("created_at")),
            updated_at=_optional_str(conversation.get("updated_at")),
            title=_optional_str(conversation.get("title")),
            workspace=_optional_str(conversation.get("workspace")),
            content=full_content,
        ),
        spans=spans,
    )


def _message_spans(source: SpanSource, span_index: int) -> list[RawArtifactSpan]:
    spans: list[RawArtifactSpan] = []
    chunk_start = 0
    current_span_index = span_index
    while chunk_start < len(source.text):
        chunk_end = min(len(source.text), chunk_start + RAW_SPAN_MAX_CHARS)
        absolute_start = source.start_offset + chunk_start
        absolute_end = source.start_offset + chunk_end
        spans.append(
            RawArtifactSpan(
                id=_span_id(
                    source.artifact_id,
                    current_span_index,
                    source.message_id,
                    absolute_start,
                    absolute_end,
                ),
                artifact_id=source.artifact_id,
                span_index=current_span_index,
                message_index=source.message_index,
                message_id=source.message_id,
                role=source.role,
                created_at=source.created_at,
                start_offset=absolute_start,
                end_offset=absolute_end,
                content=source.text[chunk_start:chunk_end],
            )
        )
        current_span_index += 1
        if chunk_end == len(source.text):
            break
        chunk_start = max(chunk_end - RAW_SPAN_OVERLAP_CHARS, chunk_start + 1)
    return spans


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
    source_path: str,
) -> str:
    stable_key = f"{provider}\0{source_conversation_id}\0{source_path}"
    return hashlib.sha256(stable_key.encode("utf-8")).hexdigest()


def _span_id(
    artifact_id: str,
    span_index: int,
    message_id: str,
    start_offset: int,
    end_offset: int,
) -> str:
    stable_key = f"{artifact_id}\0{span_index}\0{message_id}\0{start_offset}\0{end_offset}"
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
