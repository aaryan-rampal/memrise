# memrise

A small local memory daemon for agent workflows.

## v0

- SQLite-backed memory store.
- CLI-first interface.
- Agent guidance printed by default, with `--no-help` for raw output.
- Constrained v0 `source` and `memory_type` values.
- SQLite FTS5 search.
- Canonical raw chat artifacts indexed separately from curated memories.
- Embedding code paths are temporarily disabled while retrieval quality is reviewed.

## Setup

```bash
uv pip install --python .venv/bin/python -e .
```

## Usage

```bash
mem add "The operator wants small dogfoodable loops for local infrastructure."
mem search "dogfoodable loops"
mem recent
mem update <id> "Updated self-contained memory text."
mem delete <id>
```

Use a specific database file:

```bash
mem --db ~/.local/share/memrise/memories.sqlite3 add "A memory"
```

Search only curated memories, raw artifacts, or both:

```bash
mem search "deployment notes" --scope memories
mem search "deployment notes" --scope raw
mem search "deployment notes" --scope both
```

Raw artifact search results are JSONL. Each line includes the raw artifact id,
provider metadata, match bounds, snippet window bounds, and a bounded snippet.
Use `raw show` with explicit bounds to inspect the source text:

```bash
mem search "deployment notes" --scope raw
mem raw show <raw-id> --start 0 --end 500
```

Full raw artifact output requires an explicit boolean:

```bash
mem raw show <raw-id> --full true
```

Allowed sources:

```text
chatgpt, claude, opencode, codex, claude-code
```

Allowed memory types:

```text
preference, fact, project_context, decision, pattern, warning
```

## Raw Chats

Raw chat conversion is a two-step process:

```bash
MEMORIES_CLAUDE_EXPORT_DIR=~/Downloads/claude-export mem raw-chats link-sources
mem raw-chats canonicalize
mem raw-chats index
```

The default source layout is:

```text
data/raw/chats/
  claude-export -> $MEMORIES_CLAUDE_EXPORT_DIR
  claude-code -> ~/.claude/projects
  codex -> ~/.codex/sessions
  opencode -> ~/.local/share/opencode/storage
```

Canonical chats are written to `data/canonical/chats/*.jsonl`. `data/` is ignored
because it can contain private raw text. Claude Code `~/.claude/transcripts` is
not canonical for this project and is intentionally excluded.

Tool calls and tool results are sanitized in canonical chat output. The system
records that a tool call happened; it does not preserve raw arguments, command
output, or tool result bodies.

## Embeddings

Embedding-based writes and searches are currently disabled:

```bash
mem semantic-search "query"
mem --embedder openrouter add "A memory"
mem raw-chats index --embed
```

Each command exits non-zero with an explicit temporary-disable message. Existing
embedding rows can be removed idempotently:

```bash
mem embeddings clear
```
