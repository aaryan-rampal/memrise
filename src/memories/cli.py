from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from loguru import logger

from memories.embedder import OpenRouterEmbedder
from memories.logging import configure_logging
from memories.models import (
    AddMemory,
    Memory,
    MemoryType,
    RawArtifact,
    RawArtifactSearchMatch,
    RawArtifactSpan,
    Source,
)
from memories.raw_artifacts import RawArtifactIndexOptions, index_canonical_chats
from memories.raw_chats import ensure_raw_chat_links, write_canonical_chats
from memories.service import MemoryService
from memories.sqlite_store import SQLiteMemoryStore

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDINGS_DISABLED_MESSAGE = (
    "embedding features are temporarily disabled; use lexical search while embeddings are reviewed"
)
RAW_SNIPPET_CONTEXT_CHARS = 80

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from memories.embedder import Embedder, HttpPost


@dataclass(frozen=True)
class RawSnippet:
    match_start: int
    match_end: int
    match_term: str
    window_start: int
    window_end: int
    text: str


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    environ: Mapping[str, str] | None = None,
    post_json: HttpPost | None = None,
) -> int:
    """Run the `mem` command-line interface."""
    parser = _parser()
    args = parser.parse_args(argv)
    configure_logging(stderr, level=(environ or os.environ).get("MEMORIES_LOG_LEVEL", "INFO"))
    store = SQLiteMemoryStore(args.db)
    if args.command == "embeddings":
        try:
            _dispatch_embeddings(args, store, stdout)
        except ValueError as error:
            print(f"error: {error}", file=stderr)
            return 1
        return 0
    if args.command == "raw-chats":
        try:
            _reject_embedding_flags(args)
            _dispatch_raw_chats(
                args,
                store,
                stdout,
                environ or os.environ,
                post_json,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            print(f"error: {error}", file=stderr)
            return 1
        return 0
    store.initialize()
    try:
        _reject_embedding_flags(args)
        service = MemoryService(
            store,
            embedder=_embedder_from_args(args, environ or os.environ, post_json),
        )
        _print_guidance(args, stdout)
        _dispatch(args, service, store, stdout)
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"error: {error}", file=stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mem")
    parser.add_argument("--db", type=Path, default=Path("memories.sqlite3"))
    parser.add_argument("--no-help", action="store_true")
    parser.add_argument("--embedder", choices=["none", "openrouter"], default="none")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimensions", type=int, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add")
    add.add_argument("content")
    _metadata_args(add)

    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument(
        "--scope",
        choices=["auto", "memories", "raw", "both"],
        default="auto",
    )

    semantic_search = subparsers.add_parser("semantic-search")
    semantic_search.add_argument("query")
    semantic_search.add_argument("--limit", type=int, default=8)
    semantic_search.add_argument(
        "--scope",
        choices=["auto", "memories", "raw", "both"],
        default="auto",
    )

    embeddings = subparsers.add_parser("embeddings")
    embeddings_subparsers = embeddings.add_subparsers(
        dest="embeddings_command",
        required=True,
    )
    embeddings_subparsers.add_parser("clear")

    _curated_parser(subparsers)

    recent = subparsers.add_parser("recent")
    recent.add_argument("--limit", type=int, default=8)

    update = subparsers.add_parser("update")
    update.add_argument("id")
    update.add_argument("content")
    _metadata_args(update)

    delete = subparsers.add_parser("delete")
    delete.add_argument("id")

    raw_chats = subparsers.add_parser("raw-chats")
    raw_subparsers = raw_chats.add_subparsers(dest="raw_chats_command", required=True)

    link_sources = raw_subparsers.add_parser("link-sources")
    link_sources.add_argument("--raw-dir", type=Path, default=Path("data/raw/chats"))

    canonicalize = raw_subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--raw-dir", type=Path, default=Path("data/raw/chats"))
    canonicalize.add_argument("--output-dir", type=Path, default=Path("data/canonical/chats"))
    canonicalize.add_argument(
        "--provider",
        dest="providers",
        action="append",
        choices=["claude-export", "claude-code", "codex", "opencode"],
        default=None,
    )

    index = raw_subparsers.add_parser("index")
    index.add_argument("--canonical-dir", type=Path, default=Path("data/canonical/chats"))
    index.add_argument(
        "--provider",
        dest="providers",
        action="append",
        choices=["claude-export", "claude-code", "codex", "opencode"],
        default=None,
    )
    index.add_argument("--embed", action="store_true")
    index.add_argument("--rebuild-embeddings", action="store_true")

    raw = subparsers.add_parser("raw")
    raw_subparsers = raw.add_subparsers(dest="raw_command", required=True)

    raw_show = raw_subparsers.add_parser("show")
    raw_show.add_argument("id")
    raw_show.add_argument("--start", type=int, default=None)
    raw_show.add_argument("--end", type=int, default=None)
    raw_show.add_argument("--full", choices=["true", "false"], default="false")
    return parser


def _curated_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    curated = subparsers.add_parser("curated")
    curated_subparsers = curated.add_subparsers(dest="curated_command", required=True)
    curated_import = curated_subparsers.add_parser("import")
    curated_import.add_argument("--input", type=Path, default=Path("data/curated/memories.jsonl"))


def _metadata_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default="codex")
    parser.add_argument("--type", dest="memory_type", default="fact")
    parser.add_argument("--importance", type=float, default=0.5)


def _embedder_from_args(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    post_json: HttpPost | None,
) -> Embedder | None:
    if args.embedder == "none":
        return None
    api_key = environ.get("OPENROUTER_API_KEY", "")
    return OpenRouterEmbedder(
        api_key=api_key,
        model=args.embedding_model or DEFAULT_EMBEDDING_MODEL,
        dimensions=args.embedding_dimensions,
        post_json=post_json,
    )


def _reject_embedding_flags(args: argparse.Namespace) -> None:
    if args.command == "semantic-search":
        raise ValueError(EMBEDDINGS_DISABLED_MESSAGE)
    if args.embedder != "none":
        raise ValueError(EMBEDDINGS_DISABLED_MESSAGE)
    if getattr(args, "embed", False):
        raise ValueError(EMBEDDINGS_DISABLED_MESSAGE)


def _dispatch_embeddings(
    args: argparse.Namespace,
    store: SQLiteMemoryStore,
    stdout: TextIO,
) -> None:
    store.initialize()
    if args.embeddings_command == "clear":
        memory_count, raw_count = store.clear_all_embeddings()
        print(f"memory-embeddings-cleared\t{memory_count}", file=stdout)
        print(f"raw-embeddings-cleared\t{raw_count}", file=stdout)


def _print_guidance(args: argparse.Namespace, stdout: TextIO) -> None:
    if args.no_help:
        return
    guidance = {
        "add": "Adds one self-contained memory. Use --no-help to hide this guidance.",
        "search": "Searches saved memories with SQLite FTS5. Use --no-help to hide this guidance.",
        "semantic-search": (
            "Searches saved memories by embedding similarity. Use --no-help to hide this guidance."
        ),
        "recent": "Lists newest memories first. Use --no-help to hide this guidance.",
        "update": "Updates one memory by id. Use --no-help to hide this guidance.",
        "delete": "Deletes one memory by id. Use --no-help to hide this guidance.",
        "curated": "Imports curated memories from JSONL. Use --no-help to hide this guidance.",
        "raw-chats": (
            "Links and canonicalizes raw chat exports. Use --no-help to hide this guidance."
        ),
        "raw": "Reads indexed raw artifacts. Use --no-help to hide this guidance.",
    }
    print(guidance[args.command], file=stdout)


def _dispatch(
    args: argparse.Namespace,
    service: MemoryService,
    store: SQLiteMemoryStore,
    stdout: TextIO,
) -> None:
    if args.command == "add":
        memory = service.add_memory(
            AddMemory(
                content=args.content,
                source=args.source,
                memory_type=args.memory_type,
                importance=args.importance,
            )
        )
        print(f"added {memory.id}", file=stdout)
    elif args.command == "search":
        _print_search(args, service, store, stdout)
    elif args.command == "semantic-search":
        _print_semantic_search(args, service, store, stdout)
    elif args.command == "recent":
        _print_memories(service.recent(args.limit), stdout)
    elif args.command == "update":
        memory = service.update_memory(
            args.id,
            content=args.content,
            source=args.source,
            memory_type=args.memory_type,
            importance=args.importance,
        )
        print(f"updated {memory.id}", file=stdout)
    elif args.command == "delete":
        deleted = service.delete_memory(args.id)
        print(f"deleted {args.id}" if deleted else f"missing {args.id}", file=stdout)
    elif args.command == "curated":
        _dispatch_curated(args, store, stdout)
    elif args.command == "raw":
        _dispatch_raw(args, store, stdout)


def _dispatch_curated(
    args: argparse.Namespace,
    store: SQLiteMemoryStore,
    stdout: TextIO,
) -> None:
    if args.curated_command == "import":
        memories = _curated_memories_from_jsonl(args.input)
        store.upsert_memories(memories)
        print(f"curated-imported\t{len(memories)}", file=stdout)


def _curated_memories_from_jsonl(path: Path) -> list[Memory]:
    timestamp = int(time.time())
    memories: list[Memory] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                memories.append(_curated_memory_from_json(line, path, line_number, timestamp))
    return memories


def _curated_memory_from_json(
    line: str,
    path: Path,
    line_number: int,
    timestamp: int,
) -> Memory:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        msg = f"{path}:{line_number}: invalid JSON: {error.msg}"
        raise ValueError(msg) from error
    if not isinstance(payload, dict):
        msg = f"{path}:{line_number}: curated memory must be a JSON object"
        raise TypeError(msg)
    add_memory = AddMemory(
        content=_required_string(payload, "content", path, line_number),
        source=str(payload.get("source", "codex")),
        memory_type=str(payload.get("memory_type", "fact")),
        importance=float(payload.get("importance", 0.5)),
    )
    source = Source.parse(add_memory.source)
    memory_type = MemoryType.parse(add_memory.memory_type)
    return Memory(
        id=_curated_memory_id(payload, add_memory),
        content=add_memory.content,
        source=source,
        memory_type=memory_type,
        importance=add_memory.importance,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _required_string(
    payload: dict[object, object],
    field: str,
    path: Path,
    line_number: int,
) -> str:
    value = payload.get(field)
    if isinstance(value, str):
        return value
    msg = f"{path}:{line_number}: '{field}' must be a string"
    raise ValueError(msg)


def _curated_memory_id(payload: dict[object, object], memory: AddMemory) -> str:
    raw_id = payload.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    stable_payload = {
        "content": memory.content,
        "source": Source.parse(memory.source).value,
        "memory_type": MemoryType.parse(memory.memory_type).value,
    }
    encoded = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dispatch_raw(
    args: argparse.Namespace,
    store: SQLiteMemoryStore,
    stdout: TextIO,
) -> None:
    if args.raw_command == "show":
        artifact = store.get_raw_artifact(args.id)
        if artifact is None:
            msg = f"Raw artifact '{args.id}' does not exist"
            raise ValueError(msg)
        start, end = _raw_show_bounds(args, len(artifact.content))
        print(f"raw\t{artifact.id}\t{start}:{end}\t{artifact.content[start:end]}", file=stdout)


def _raw_show_bounds(args: argparse.Namespace, content_length: int) -> tuple[int, int]:
    if args.full == "true":
        return 0, content_length
    if args.start is None or args.end is None:
        msg = "raw show requires bounded output; pass --full true or provide --start and --end"
        raise ValueError(msg)
    if args.start < 0:
        msg = "--start must be greater than or equal to 0"
        raise ValueError(msg)
    if args.end <= args.start:
        msg = "--end must be greater than --start"
        raise ValueError(msg)
    if args.end > content_length:
        msg = f"--end must be less than or equal to content length {content_length}"
        raise ValueError(msg)
    return args.start, args.end


def _dispatch_raw_chats(
    args: argparse.Namespace,
    store: SQLiteMemoryStore,
    stdout: TextIO,
    environ: Mapping[str, str],
    post_json: HttpPost | None,
) -> None:
    if args.raw_chats_command == "link-sources":
        logger.info("raw_chat_link_sources_start raw_dir={}", args.raw_dir)
        links = ensure_raw_chat_links(args.raw_dir)
        for provider, path in links.items():
            logger.info("raw_chat_link provider={} path={}", provider, path)
            print(f"{provider}\t{path}", file=stdout)
        logger.info("raw_chat_link_sources_complete count={}", len(links))
    elif args.raw_chats_command == "canonicalize":
        logger.info(
            "raw_chat_canonicalize_start raw_dir={} output_dir={} providers={}",
            args.raw_dir,
            args.output_dir,
            args.providers or "all",
        )
        counts = write_canonical_chats(
            args.raw_dir,
            args.output_dir,
            providers=args.providers,
        )
        for provider, count in counts.items():
            logger.info(
                "raw_chat_canonicalize_provider provider={} conversations={}", provider, count
            )
            print(f"{provider}\t{count}", file=stdout)
        logger.info("raw_chat_canonicalize_complete providers={}", len(counts))
    elif args.raw_chats_command == "index":
        store.initialize()
        if args.embed and args.embedding_model is None:
            msg = "raw chat embedding requires explicit --embedding-model"
            raise ValueError(msg)
        embedder = _embedder_from_args(args, environ, post_json) if args.embed else None
        if args.embed and embedder is None:
            msg = "raw chat embedding requires --embedder openrouter"
            raise ValueError(msg)
        count = index_canonical_chats(
            store,
            args.canonical_dir,
            RawArtifactIndexOptions(
                providers=args.providers,
                embedder=embedder,
                rebuild_embeddings=args.rebuild_embeddings,
            ),
        )
        logger.info("raw_chat_index_complete artifacts={}", count)
        if args.embed:
            print(f"raw-embeddings\t{count}", file=stdout)


def _print_search(
    args: argparse.Namespace,
    service: MemoryService,
    store: SQLiteMemoryStore,
    stdout: TextIO,
) -> None:
    scope = _resolved_scope(args.scope, args.query, store)
    if scope == "memories":
        _print_memories(service.search(args.query, args.limit), stdout)
        return
    if scope == "raw":
        _print_raw_artifacts(store.search_raw_artifacts(args.query, args.limit), args.query, stdout)
        return
    _print_memory_hits(service.search(args.query, args.limit), stdout)
    _print_raw_artifacts(store.search_raw_artifacts(args.query, args.limit), args.query, stdout)


def _print_semantic_search(
    args: argparse.Namespace,
    service: MemoryService,
    store: SQLiteMemoryStore,
    stdout: TextIO,
) -> None:
    scope = _resolved_scope(args.scope, args.query, store)
    query_embedding = service.embed_query(args.query)
    if scope == "memories":
        _print_scored_memories(store.semantic_search(query_embedding, args.limit), stdout)
        return
    if scope == "raw":
        _print_scored_raw_artifacts(
            store.semantic_search_raw_artifacts(query_embedding, args.limit),
            stdout,
        )
        return
    _print_scored_memory_hits(store.semantic_search(query_embedding, args.limit), stdout)
    _print_scored_raw_artifacts(
        store.semantic_search_raw_artifacts(query_embedding, args.limit),
        stdout,
    )


def _resolved_scope(scope: str, query: str, store: SQLiteMemoryStore) -> str:
    if scope != "auto":
        return scope
    if store.raw_artifact_count() == 0:
        return "memories"
    normalized = query.lower()
    if any(term in normalized for term in ("raw", "transcript", "chat", "tool", "session")):
        return "raw"
    if any(term in normalized for term in ("memory", "preference", "decision", "remember")):
        return "memories"
    return "both"


def _print_memories(memories: Sequence[Memory], stdout: TextIO) -> None:
    for memory in memories:
        print(
            f"{memory.id}\t{memory.source.value}\t{memory.memory_type.value}\t"
            f"{memory.importance:.2f}\t{memory.content}",
            file=stdout,
        )


def _print_memory_hits(memories: Sequence[Memory], stdout: TextIO) -> None:
    for memory in memories:
        print(
            f"memory\t{memory.id}\t{memory.source.value}\t{memory.memory_type.value}\t"
            f"{memory.importance:.2f}\t{memory.content}",
            file=stdout,
        )


def _print_scored_memories(memories: Sequence[tuple[Memory, float]], stdout: TextIO) -> None:
    for memory, score in memories:
        print(
            f"{memory.id}\t{score:.4f}\t{memory.source.value}\t{memory.memory_type.value}\t"
            f"{memory.importance:.2f}\t{memory.content}",
            file=stdout,
        )


def _print_scored_memory_hits(memories: Sequence[tuple[Memory, float]], stdout: TextIO) -> None:
    for memory, score in memories:
        print(
            f"memory\t{memory.id}\t{score:.4f}\t{memory.source.value}\t"
            f"{memory.memory_type.value}\t{memory.importance:.2f}\t{memory.content}",
            file=stdout,
        )


def _print_raw_artifacts(
    artifacts: Sequence[RawArtifactSearchMatch],
    query: str,
    stdout: TextIO,
) -> None:
    for match in artifacts:
        snippet = _raw_search_match_snippet(match, query)
        print(json.dumps(_raw_match_json(match, snippet)), file=stdout)


def _print_scored_raw_artifacts(
    artifacts: Sequence[tuple[RawArtifact, float]],
    stdout: TextIO,
) -> None:
    for artifact, score in artifacts:
        snippet = _raw_artifact_snippet(artifact.content, "")
        raw_match = _raw_match_json(_raw_artifact_full_match(artifact), snippet)
        raw_match["score"] = round(score, 4)
        print(
            json.dumps(raw_match),
            file=stdout,
        )


def _raw_match_json(
    match: RawArtifactSearchMatch,
    snippet: RawSnippet,
) -> dict[str, object]:
    artifact = match.artifact
    span = match.span
    return {
        "kind": "raw_match",
        "artifact_id": artifact.id,
        "provider": artifact.provider,
        "conversation_id": artifact.source_conversation_id,
        "message": {
            "id": span.message_id,
            "role": span.role,
            "index": span.message_index,
        },
        "match": {
            "start": snippet.match_start,
            "end": snippet.match_end,
            "term": snippet.match_term,
        },
        "window": {"start": snippet.window_start, "end": snippet.window_end},
        "snippet": snippet.text,
    }


def _raw_artifact_full_match(artifact: RawArtifact) -> RawArtifactSearchMatch:
    return RawArtifactSearchMatch(
        artifact=artifact,
        span=RawArtifactSpan(
            id=f"{artifact.id}:full",
            artifact_id=artifact.id,
            span_index=0,
            message_index=0,
            message_id="",
            role="unknown",
            created_at=artifact.created_at,
            start_offset=0,
            end_offset=len(artifact.content),
            content=artifact.content,
        ),
    )


def _raw_search_match_snippet(match: RawArtifactSearchMatch, query: str) -> RawSnippet:
    relative_start, relative_end, match_term = _first_query_match(match.span.content, query)
    if match_term:
        match_start = match.span.start_offset + relative_start
        match_end = match.span.start_offset + relative_end
    else:
        match_start = match.span.start_offset
        match_end = match.span.start_offset
    return _raw_snippet_at_bounds(
        match.artifact.content,
        match_start,
        match_end,
        match_term,
    )


def _raw_artifact_snippet(content: str, query: str) -> RawSnippet:
    match_start, match_end, match_term = _first_query_match(content, query)
    return _raw_snippet_at_bounds(content, match_start, match_end, match_term)


def _raw_snippet_at_bounds(
    content: str,
    match_start: int,
    match_end: int,
    match_term: str,
) -> RawSnippet:
    midpoint = max(match_start, 0)
    window_start = max(0, midpoint - RAW_SNIPPET_CONTEXT_CHARS)
    window_end = min(len(content), max(match_end, midpoint) + RAW_SNIPPET_CONTEXT_CHARS)
    snippet = _one_line_text(content[window_start:window_end])
    return RawSnippet(
        match_start=max(match_start, 0),
        match_end=max(match_end, 0),
        match_term=match_term,
        window_start=window_start,
        window_end=window_end,
        text=snippet,
    )


def _first_query_match(content: str, query: str) -> tuple[int, int, str]:
    lowered_content = content.lower()
    matches: list[tuple[int, int, str]] = []
    for term in _query_terms(query):
        start = lowered_content.find(term.lower())
        if start >= 0:
            matches.append((start, start + len(term), term))
    if not matches:
        return 0, 0, ""
    return min(matches, key=lambda match: match[0])


def _query_terms(query: str) -> list[str]:
    return [term.strip('"') for term in query.split() if term.strip('"')]


def _one_line_text(text: str) -> str:
    return " ".join(text.split())


if __name__ == "__main__":
    raise SystemExit(main())
