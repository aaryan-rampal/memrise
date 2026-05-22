# memories

A small personal memory daemon for agent workflows.

## v0

- SQLite-backed memory store.
- CLI-first interface.
- Agent guidance printed by default, with `--no-help` for raw output.
- Constrained v0 `source` and `memory_type` values.
- SQLite FTS5 search.
- Swappable embedder interface with an OpenRouter implementation.

## Setup

```bash
uv pip install --python .venv/bin/python -e .
```

## Usage

```bash
mem add "Aaryan wants small dogfoodable loops for personal infrastructure."
mem search "dogfoodable loops"
mem recent
mem update <id> "Updated self-contained memory text."
mem delete <id>
```

Use a specific database file:

```bash
mem --db ~/.local/share/memories/memories.sqlite3 add "A memory"
```

Use OpenRouter embeddings on write:

```bash
OPENROUTER_API_KEY=... mem --embedder openrouter add "A memory"
```

Allowed sources:

```text
chatgpt, claude, opencode, codex, claude-code
```

Allowed memory types:

```text
preference, fact, project_context, decision, pattern, warning
```
