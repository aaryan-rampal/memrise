---
name: memrise
description: Use when extracting, searching, importing, rebuilding, or inspecting memories through the local mem CLI.
---

# Memrise CLI

## Purpose

Use this skill when the task is to work with Aaryan's local memory CLI as a
consumer: search memories, inspect raw chat evidence, rebuild the local index,
or import curated memories.

Do not treat this as repo-development guidance. If changing the codebase, follow
the repo's `AGENTS.md` and project tooling instead.

## Operating Rules

- Treat this directory as READ-ONLY.
- **ONLY** use the CLI. DO NOT RUN SCRIPTS OR MANUALLY ACCESS SQLITE.
- Treat `data/` and `memories.sqlite3` as private generated state.
- Do not use embedding commands; embedding-based paths are disabled.
- If raw output is too large, use snippets or bounded `raw show` ranges instead
  of dumping full artifacts.

## Command Form

Run commands from `/Users/aaryanrampal/personal/memories`:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help <command>
```

Use `.venv`; do not use system Python. If `.venv` is missing, stop and ask
Aaryan how to set up the environment.

## Search

Search curated memories:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help search "query" --scope memories
```

Search raw chat artifacts:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help search "query" --scope raw
```

Search both:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help search "query" --scope both
```

Inspect a raw artifact with explicit bounds:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw show <raw-id> --start 0 --end 500
```

Full raw output requires an explicit boolean:

```bash
PYTHONPATH=src .venv/bin/python -m memories.cli --no-help raw show <raw-id> --full true
```
