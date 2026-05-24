import json
from pathlib import Path

from memories.raw_chats import (
    DEFAULT_RAW_CHAT_SOURCES,
    ensure_raw_chat_links,
    write_canonical_chats,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_default_claude_code_source_excludes_transcripts() -> None:
    assert DEFAULT_RAW_CHAT_SOURCES["claude-code"] == Path.home() / ".claude" / "projects"
    assert Path.home() / ".claude" / "transcripts" not in DEFAULT_RAW_CHAT_SOURCES.values()


def test_ensure_raw_chat_links_creates_only_requested_provider_links(tmp_path: Path) -> None:
    claude_code_source = tmp_path / "claude-projects"
    codex_source = tmp_path / "codex-sessions"
    claude_code_source.mkdir()
    codex_source.mkdir()

    created = ensure_raw_chat_links(
        tmp_path / "raw" / "chats",
        sources={
            "claude-code": claude_code_source,
            "codex": codex_source,
        },
    )

    assert created == {
        "claude-code": tmp_path / "raw" / "chats" / "claude-code",
        "codex": tmp_path / "raw" / "chats" / "codex",
    }
    assert (tmp_path / "raw" / "chats" / "claude-code").resolve() == claude_code_source
    assert not (tmp_path / "raw" / "chats" / "claude-transcripts").exists()


def test_claude_export_canonicalization_ignores_conversation_summary(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "chats"
    output_dir = tmp_path / "canonical" / "chats"
    claude_source = raw_dir / "claude-export"
    claude_source.mkdir(parents=True)
    (claude_source / "conversations.json").write_text(
        json.dumps(
            [
                {
                    "uuid": "conversation-1",
                    "name": "Raw source",
                    "summary": "synthesized conclusion that must not be stored",
                    "created_at": "2026-05-24T01:00:00Z",
                    "updated_at": "2026-05-24T01:10:00Z",
                    "chat_messages": [
                        {
                            "uuid": "message-1",
                            "parent_message_uuid": None,
                            "sender": "human",
                            "created_at": "2026-05-24T01:01:00Z",
                            "text": "raw human text",
                            "content": [],
                            "attachments": [],
                            "files": [],
                        }
                    ],
                }
            ]
        )
    )

    counts = write_canonical_chats(raw_dir, output_dir, providers=["claude-export"])

    conversations = read_jsonl(output_dir / "claude-export.jsonl")
    assert counts == {"claude-export": 1}
    assert conversations[0]["title"] == "Raw source"
    assert "synthesized conclusion" not in json.dumps(conversations)
    assert conversations[0]["messages"] == [
        {
            "id": "message-1",
            "parent_id": None,
            "role": "user",
            "created_at": "2026-05-24T01:01:00Z",
            "content": [{"type": "text", "text": "raw human text"}],
            "raw": {"provider_record_type": "chat_message"},
        }
    ]


def test_codex_tool_calls_do_not_preserve_arguments_or_results(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "chats"
    output_dir = tmp_path / "canonical" / "chats"
    codex_day = raw_dir / "codex" / "2026" / "05" / "24"
    codex_day.mkdir(parents=True)
    rollout = codex_day / "rollout-2026-05-24T00-00-00-abc.jsonl"
    records = [
        {
            "timestamp": "2026-05-24T00:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": "session-1",
                "cwd": "/workspace/project",
                "timestamp": "2026-05-24T00:00:00.000Z",
            },
        },
        {
            "timestamp": "2026-05-24T00:00:01.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-1",
                "name": "exec_command",
                "arguments": '{"cmd":"secret raw command"}',
            },
        },
        {
            "timestamp": "2026-05-24T00:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "secret raw tool output",
            },
        },
    ]
    rollout.write_text("\n".join(json.dumps(record) for record in records))

    write_canonical_chats(raw_dir, output_dir, providers=["codex"])

    [conversation] = read_jsonl(output_dir / "codex.jsonl")
    assert conversation["source_conversation_id"] == "session-1"
    assert conversation["workspace"] == "/workspace/project"
    assert conversation["messages"] == [
        {
            "id": "call-1",
            "parent_id": None,
            "role": "tool",
            "created_at": "2026-05-24T00:00:01.000Z",
            "content": [{"type": "tool_call", "name": "exec_command", "status": "called"}],
            "raw": {"provider_record_type": "function_call"},
        },
        {
            "id": "call-1:result",
            "parent_id": "call-1",
            "role": "tool",
            "created_at": "2026-05-24T00:00:02.000Z",
            "content": [{"type": "tool_result", "status": "completed"}],
            "raw": {"provider_record_type": "function_call_output"},
        },
    ]
    serialized = json.dumps(conversation)
    assert "secret raw command" not in serialized
    assert "secret raw tool output" not in serialized
