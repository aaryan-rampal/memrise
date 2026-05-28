# Repository Guidance

This file is for agents developing inside this repository. For agents using the
memory CLI as a consumer, use `SKILL.md` instead.

## Project Boundaries

- Treat `data/` as generated private state.
- Do not commit raw exports, canonical chat JSONL, symlinks, or SQLite files
  containing private data.
- Keep curated memories and raw artifacts separate. Curated memories are
  self-contained facts/preferences; raw artifacts are canonical chat messages.
- Do not use `~/.claude/transcripts` as a canonical Claude Code source. Use
  `~/.claude/projects`.
- Preserve tool-call privacy. Canonical chat output may record that a tool call
  or tool result occurred, but must not store raw arguments, command output, or
  tool result text.
- Embedding CLI paths are temporarily disabled. Do not add ingestion or search
  code that calls embedding providers unless the product decision changes.

## Python Environment

Use the project environment and direct source execution:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help search "query"
```

Do not run `python -m memories.cli` without `PYTHONPATH=src` unless the package
has just been installed into `.venv`.

## Raw Chat Pipeline

```bash
MEMORIES_CLAUDE_EXPORT_DIR=/Users/aaryanrampal/personal/hindsight-setup/ai_chats/claude/data/claude \
  PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats link-sources
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats canonicalize
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats index
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help curated import \
  --input data/curated/memories.jsonl
```

To recreate the local SQLite store from generated canonical data, hard-delete
only the repo-local `memories.sqlite3`, then rerun canonicalization, indexing,
and curated import:

```bash
rm memories.sqlite3
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats canonicalize
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw-chats index
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help curated import \
  --input data/curated/memories.jsonl
```

If raw search output is too large, change presentation code to emit snippets
rather than storing less provenance.

## Verification

Run relevant checks before claiming development work is complete:

```bash
PYTHONPATH=src .venv/bin/python -m ruff check .
PYTHONPATH=src .venv/bin/python -m ruff format --check .
PYTHONPATH=src .venv/bin/python -m ty check
PYTHONPATH=src .venv/bin/python -m pytest -q
```
