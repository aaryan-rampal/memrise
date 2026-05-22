import io
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
