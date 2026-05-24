import io
import json
from pathlib import Path

from memories.cli import main
from memories.embedder import Embedding, HttpPost
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
            "Aaryan prefers explicit verification before completion claims.",
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
            "Aaryan wants SQLite as the v0 production store.",
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
            "Aaryan wants hybrid retrieval after SQLite FTS5 proves useful.",
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


def test_add_can_store_openrouter_embedding(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    def post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        del url, payload, headers, timeout
        return {
            "data": [{"embedding": [0.4, 0.5]}],
            "model": "openai/text-embedding-3-small",
        }

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "--embedder",
            "openrouter",
            "add",
            "Aaryan can use OpenRouter embeddings now and local embeddings later.",
        ],
        environ={"OPENROUTER_API_KEY": "secret"},
        post_json=post_json,
    )
    memory_id = stdout.split()[1]
    store = SQLiteMemoryStore(db_path)
    store.initialize()

    assert exit_code == 0
    assert stderr == ""
    assert store.get_embedding(memory_id) == Embedding(
        provider="openrouter",
        model="openai/text-embedding-3-small",
        vector=[0.4, 0.5],
    )


def test_semantic_search_uses_embedder_for_query(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    def post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, object]:
        del url, headers, timeout
        vectors = {
            "Aaryan wants careful git recovery.": [1.0, 0.0],
            "Aaryan wants Spotify weekly mixes.": [0.0, 1.0],
            "recover lost commit": [0.9, 0.1],
        }
        return {
            "data": [{"embedding": vectors[str(payload["input"])]}],
            "model": "fake/embedding",
        }

    run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "--embedder",
            "openrouter",
            "add",
            "Aaryan wants careful git recovery.",
        ],
        environ={"OPENROUTER_API_KEY": "secret"},
        post_json=post_json,
    )
    run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "--embedder",
            "openrouter",
            "add",
            "Aaryan wants Spotify weekly mixes.",
        ],
        environ={"OPENROUTER_API_KEY": "secret"},
        post_json=post_json,
    )

    exit_code, stdout, stderr = run_cli(
        [
            "--db",
            str(db_path),
            "--no-help",
            "--embedder",
            "openrouter",
            "semantic-search",
            "recover lost commit",
            "--limit",
            "1",
        ],
        environ={"OPENROUTER_API_KEY": "secret"},
        post_json=post_json,
    )

    assert exit_code == 0
    assert stderr == ""
    assert "careful git recovery" in stdout
    assert "Spotify weekly mixes" not in stdout
    assert "0." in stdout


def test_semantic_search_requires_embedder(tmp_path: Path) -> None:
    db_path = tmp_path / "memories.sqlite3"

    exit_code, stdout, stderr = run_cli(
        ["--db", str(db_path), "--no-help", "semantic-search", "recover lost commit"]
    )

    assert exit_code == 1
    assert stdout == ""
    assert "semantic search requires an embedder" in stderr


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
    assert stderr == ""
    assert "codex\t1" in stdout
    assert (output_dir / "codex.jsonl").exists()
