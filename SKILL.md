---
name: memories
description: Use when working in this repository on the local memory CLI, raw chat canonicalization, raw artifact indexing, or retrieval behavior.
---

# Memories

## Workflow

1. Treat `data/` as generated private state. Do not commit raw exports,
   canonical chat JSONL, symlinks, or SQLite files containing private data.
2. Keep curated memories and raw artifacts separate. Curated memories are
   self-contained facts/preferences; raw artifacts are canonical chat messages.
3. Do not use `~/.claude/transcripts` as a canonical Claude Code source. Use
   `~/.claude/projects`.
4. Preserve tool-call privacy. Canonical chat output may say a tool call or tool
   result occurred, but must not store raw arguments, command output, or tool
   result text.
5. Embedding CLI paths are temporarily disabled. Do not add ingestion or search
   code that calls embedding providers unless the product decision changes.

## Commands

Use the project environment:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help search "query"
```

Run verification before claiming completion:

```bash
PYTHONPATH=src .venv/bin/python -m ruff check .
PYTHONPATH=src .venv/bin/python -m ruff format --check .
PYTHONPATH=src .venv/bin/python -m ty check
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Raw Chat Pipeline

```bash
MEMORIES_CLAUDE_EXPORT_DIR=~/Downloads/claude-export \
  PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats link-sources
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats canonicalize
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats index
```

If raw search output is too large, change presentation code to emit snippets
rather than storing less provenance.
