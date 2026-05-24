from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from memories.embedder import OpenRouterEmbedder
from memories.models import AddMemory
from memories.raw_chats import ensure_raw_chat_links, write_canonical_chats
from memories.service import MemoryService
from memories.sqlite_store import SQLiteMemoryStore

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from memories.embedder import Embedder, HttpPost
    from memories.models import Memory


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
    if args.command == "raw-chats":
        try:
            _dispatch_raw_chats(args, stdout)
        except (FileExistsError, FileNotFoundError, ValueError) as error:
            print(f"error: {error}", file=stderr)
            return 1
        return 0
    store = SQLiteMemoryStore(args.db)
    store.initialize()
    try:
        service = MemoryService(
            store,
            embedder=_embedder_from_args(args, environ or os.environ, post_json),
        )
        _print_guidance(args, stdout)
        _dispatch(args, service, stdout)
    except ValueError as error:
        print(f"error: {error}", file=stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mem")
    parser.add_argument("--db", type=Path, default=Path("memories.sqlite3"))
    parser.add_argument("--no-help", action="store_true")
    parser.add_argument("--embedder", choices=["none", "openrouter"], default="none")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--embedding-dimensions", type=int, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add")
    add.add_argument("content")
    _metadata_args(add)

    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)

    semantic_search = subparsers.add_parser("semantic-search")
    semantic_search.add_argument("query")
    semantic_search.add_argument("--limit", type=int, default=8)

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
    return parser


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
        model=args.embedding_model,
        dimensions=args.embedding_dimensions,
        post_json=post_json,
    )


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
        "raw-chats": (
            "Links and canonicalizes raw chat exports. Use --no-help to hide this guidance."
        ),
    }
    print(guidance[args.command], file=stdout)


def _dispatch(args: argparse.Namespace, service: MemoryService, stdout: TextIO) -> None:
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
        _print_memories(service.search(args.query, args.limit), stdout)
    elif args.command == "semantic-search":
        _print_scored_memories(service.semantic_search(args.query, args.limit), stdout)
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


def _dispatch_raw_chats(args: argparse.Namespace, stdout: TextIO) -> None:
    if args.raw_chats_command == "link-sources":
        links = ensure_raw_chat_links(args.raw_dir)
        for provider, path in links.items():
            print(f"{provider}\t{path}", file=stdout)
    elif args.raw_chats_command == "canonicalize":
        counts = write_canonical_chats(
            args.raw_dir,
            args.output_dir,
            providers=args.providers,
        )
        for provider, count in counts.items():
            print(f"{provider}\t{count}", file=stdout)


def _print_memories(memories: Sequence[Memory], stdout: TextIO) -> None:
    for memory in memories:
        print(
            f"{memory.id}\t{memory.source.value}\t{memory.memory_type.value}\t"
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


if __name__ == "__main__":
    raise SystemExit(main())
