import io
import json
from pathlib import Path

from memories.cli import main
from memories.embedder import Embedding, HttpPost
from memories.models import AddMemory, RawArtifact
from memories.sqlite_store import SQLiteMemoryStore


def run_cli(
    args: list[str],
    *,
    environ: dict[str, str] | None = None,
    post_json: HttpPost | None = None,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(args, stdout=stdout, stderr=stderr, environ=environ, post_json=post_json)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_add_prints_agent_guidance_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "add",
            "The operator prefers explicit verification before completion claims.",
            "--source",
            "codex",
            "--type",
            "preference",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "--no-help" in stdout
    assert "added" in stdout


def test_no_help_suppresses_agent_guidance(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "add",
            "The operator wants SQLite as the v0 production store.",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "--no-help" not in stdout
    assert "added" in stdout


def test_search_finds_memory_added_by_cli(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "add",
            "The operator wants hybrid retrieval after SQLite FTS5 proves useful.",
            "--type",
            "decision",
            "--importance",
            "0.9",
        ]
    )

    exit_code, stdout, stderr = run_cli(
        ["--db", str(db_path), "--no-help", "search", "hybrid retrieval"]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "hybrid retrieval" in stdout
    assert "decision" in stdout


def test_add_rejects_embedding_writes_while_disabled(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "--embedder",
            "openrouter",
            "add",
            "The operator can use OpenRouter embeddings now and local embeddings later.",
        ],
        environ={"OPENROUTER_API_KEY": "secret"},
    )

    assert exit_code == 1
    assert stdout == ""
    assert "embedding features are temporarily disabled" in stderr


def test_semantic_search_is_disabled(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "semantic-search",
            "recover lost commit",
            "--limit",
            "1",
        ],
    )

    assert exit_code == 1
    assert stdout == ""
    assert "embedding features are temporarily disabled" in stderr


def test_raw_chats_canonicalize_writes_provider_jsonl(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "chats"
    output_dir = tmp_path / "canonical" / "chats"
    codex_day = raw_dir / "codex" / "2026" / "05" / "24"
    codex_day.mkdir(parents=True)
    (codex_day / "rollout-1.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-24T00:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "session-1", "cwd": "/workspace/project"},
            }
        )
        + "\n"
    )

    exit_code, stdout, stderr = run_cli(
        [
            "--no-help",
            "raw-chats",
            "canonicalize",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "codex",
        ]
    )

    assert exit_code == 0
    assert "raw_chat_canonicalize_start" in stderr
    assert "canonical_chat_provider_complete provider=codex conversations=1" in stderr
    assert "codex\t1" in stdout
    assert (output_dir / "codex.jsonl").exists()


def test_search_auto_scope_includes_memory_and_raw_artifact_results(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
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
                        "created_at": None,
                        "content": [{"type": "text", "text": "canonical artifact detail"}],
                    }
                ],
            }
        )
        + "\n"
    )
    run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "add",
            "canonical memory detail",
        ]
    )
    run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "raw-chats",
            "index",
            "--canonical-dir",
            str(canonical_dir),
            "--provider",
            "codex",
        ]
    )

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "search",
            "canonical detail",
            "--scope",
            "auto",
        ]
    )

    assert exit_code == 0
    assert stderr == ""
    assert "memory\t" in stdout
    assert '"kind": "raw_match"' in stdout
    assert '"provider": "codex"' in stdout
    assert '"conversation_id": "session-1"' in stdout
    assert '"message_id": "message-1"' in stdout


def test_raw_search_prints_bounded_jsonl_snippet_with_id_and_position(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    store = SQLiteMemoryStore(db_path)
    store.initialize()
    store.upsert_raw_artifact(
        RawArtifact(
            id="raw-1",
            provider="codex",
            source_path="data/canonical/chats/codex.jsonl",
            source_conversation_id="session-1",
            message_id="message-1",
            role="user",
            created_at=None,
            content="alpha " * 60 + "needle centered detail " + "omega " * 60,
        )
    )

    exit_code, stdout, stderr = run_cli(
        ["--db", str(db_path), "--no-help", "search", "needle", "--scope", "raw"]
    )

    assert exit_code == 0
    assert stderr == ""
    result = json.loads(stdout)
    assert result == {
        "kind": "raw_match",
        "id": "raw-1",
        "provider": "codex",
        "conversation_id": "session-1",
        "message_id": "message-1",
        "role": "user",
        "match": {"start": 360, "end": 366, "term": "needle"},
        "window": {"start": 280, "end": 446},
        "snippet": (
            "a alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha "
            "needle centered detail omega omega omega omega omega omega omega omega omega omega ome"
        ),
    }


def test_raw_show_requires_bounds_or_explicit_full_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    store = SQLiteMemoryStore(db_path)
    store.initialize()
    store.upsert_raw_artifact(
        RawArtifact(
            id="raw-1",
            provider="codex",
            source_path="data/canonical/chats/codex.jsonl",
            source_conversation_id="session-1",
            message_id="message-1",
            role="user",
            created_at=None,
            content="raw artifact content",
        )
    )

    exit_code, stdout, stderr = run_cli(["--db", str(db_path), "--no-help", "raw", "show", "raw-1"])

    assert exit_code == 1
    assert stdout == ""
    assert "pass --full true or provide --start and --end" in stderr


def test_raw_show_prints_bounded_slice(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    store = SQLiteMemoryStore(db_path)
    store.initialize()
    store.upsert_raw_artifact(
        RawArtifact(
            id="raw-1",
            provider="codex",
            source_path="data/canonical/chats/codex.jsonl",
            source_conversation_id="session-1",
            message_id="message-1",
            role="user",
            created_at=None,
            content="0123456789abcdef",
        )
    )

    exit_code, stdout, stderr = run_cli(
        ["--db", str(db_path), "--no-help", "raw", "show", "raw-1", "--start", "2", "--end", "8"]
    )

    assert exit_code == 0
    assert stderr == ""
    assert stdout == "raw\traw-1\t2:8\t234567\n"


def test_raw_show_prints_full_content_only_with_explicit_true(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    store = SQLiteMemoryStore(db_path)
    store.initialize()
    store.upsert_raw_artifact(
        RawArtifact(
            id="raw-1",
            provider="codex",
            source_path="data/canonical/chats/codex.jsonl",
            source_conversation_id="session-1",
            message_id="message-1",
            role="user",
            created_at=None,
            content="full raw artifact content",
        )
    )

    exit_code, stdout, stderr = run_cli(
        ["--db", str(db_path), "--no-help", "raw", "show", "raw-1", "--full", "true"]
    )

    assert exit_code == 0
    assert stderr == ""
    assert stdout == "raw\traw-1\t0:25\tfull raw artifact content\n"


def test_raw_chats_index_logs_progress_to_stderr(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
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
                        "created_at": None,
                        "content": [{"type": "text", "text": "logged raw artifact detail"}],
                    }
                ],
            }
        )
        + "\n"
    )

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "raw-chats",
            "index",
            "--canonical-dir",
            str(canonical_dir),
            "--provider",
            "codex",
        ]
    )

    assert exit_code == 0
    assert stdout == ""
    assert "raw_artifact_index_start" in stderr
    assert "raw_artifact_file_complete" in stderr
    assert "raw_artifact_index_replace_complete artifacts=1" in stderr
    assert "raw_chat_index_complete artifacts=1" in stderr


def test_raw_chats_index_rejects_embedding_while_disabled(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    canonical_dir = tmp_path / "canonical" / "chats"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "opencode.jsonl").write_text(
        json.dumps(
            {
                "provider": "opencode",
                "source_path": "data/raw/chats/opencode/session.json",
                "source_conversation_id": "session-1",
                "messages": [
                    {
                        "id": "message-1",
                        "role": "assistant",
                        "created_at": None,
                        "content": [{"type": "text", "text": "raw artifact embedding text"}],
                    }
                ],
            }
        )
        + "\n"
    )

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "--embedder",
            "openrouter",
            "--embedding-model",
            "qwen/qwen3-embedding-8b",
            "raw-chats",
            "index",
            "--canonical-dir",
            str(canonical_dir),
            "--provider",
            "opencode",
            "--embed",
        ],
        environ={"OPENROUTER_API_KEY": "secret"},
    )

    assert exit_code == 1
    assert stdout == ""
    assert "embedding features are temporarily disabled" in stderr


def test_embeddings_clear_removes_memory_and_raw_artifact_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"
    store = SQLiteMemoryStore(db_path)
    store.initialize()
    memory = store.add(AddMemory(content="memory with embedding"))
    artifact = RawArtifact(
        id="raw-1",
        provider="codex",
        source_path="data/canonical/chats/codex.jsonl",
        source_conversation_id="session-1",
        message_id="message-1",
        role="user",
        created_at=None,
        content="raw artifact with embedding",
    )
    store.upsert_raw_artifact(artifact)
    store.save_embedding(memory.id, Embedding(provider="fake", model="tiny", vector=[1.0]))
    store.save_raw_artifact_embedding(
        artifact.id,
        Embedding(provider="fake", model="tiny", vector=[1.0]),
    )

    exit_code, stdout, stderr = run_cli(["--db", str(db_path), "--no-help", "embeddings", "clear"])

    assert exit_code == 0
    assert stderr == ""
    assert "memory-embeddings-cleared\t1" in stdout
    assert "raw-embeddings-cleared\t1" in stdout
    assert store.get_embedding(memory.id) is None
    assert store.raw_artifact_embedding_count() == 0
