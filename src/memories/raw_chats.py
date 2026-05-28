from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from loguru import logger

JsonObject = dict[str, Any]
ContentItems = list[JsonObject]

DEFAULT_RAW_CHAT_SOURCES = {
    "claude-export": Path(
        os.environ.get("MEMORIES_CLAUDE_EXPORT_DIR", "~/Downloads/claude-export")
    ),
    "claude-code": Path.home() / ".claude" / "projects",
    "codex": Path.home() / ".codex" / "sessions",
    "opencode": Path.home() / ".local" / "share" / "opencode" / "storage",
}


@dataclass(frozen=True)
class ConversationRecord:
    provider: str
    source_path: Path
    source_conversation_id: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    workspace: str | None
    messages: list[JsonObject]


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    parent_id: str | None
    role: str
    created_at: str | None
    content: ContentItems
    record_type: str


class RawChatAdapter(Protocol):
    """Adapter contract for converting one provider into canonical chat JSON."""

    provider: str

    def conversations(self, source_dir: Path) -> list[JsonObject]:
        """Return canonical conversation records from one source directory."""


class ClaudeExportAdapter:
    """Convert Anthropic account-export conversations into canonical chats."""

    provider = "claude-export"

    def conversations(self, source_dir: Path) -> list[JsonObject]:
        """Return canonical conversations from `conversations.json`."""
        conversations_path = source_dir / "conversations.json"
        if not conversations_path.exists():
            return []
        source_rows = _json_array(conversations_path)
        conversations: list[JsonObject] = []
        for source in source_rows:
            messages = [
                _message(
                    MessageRecord(
                        message_id=str(message.get("uuid", "")),
                        parent_id=_optional_str(message.get("parent_message_uuid")),
                        role=_claude_role(message.get("sender")),
                        created_at=_optional_str(message.get("created_at")),
                        content=_claude_message_content(message),
                        record_type="chat_message",
                    )
                )
                for message in _list_of_objects(source.get("chat_messages"))
            ]
            conversations.append(
                _conversation(
                    ConversationRecord(
                        provider=self.provider,
                        source_path=conversations_path,
                        source_conversation_id=str(source.get("uuid", "")),
                        title=_optional_str(source.get("name")),
                        created_at=_optional_str(source.get("created_at")),
                        updated_at=_optional_str(source.get("updated_at")),
                        workspace=None,
                        messages=messages,
                    )
                )
            )
        return conversations


class ClaudeCodeAdapter:
    """Convert Claude Code project JSONL sessions into canonical chats."""

    provider = "claude-code"

    def conversations(self, source_dir: Path) -> list[JsonObject]:
        """Return canonical conversations from Claude Code project transcripts."""
        return [
            self._conversation_from_file(path)
            for path in sorted(source_dir.rglob("*.jsonl"))
            if path.is_file()
        ]

    def _conversation_from_file(self, path: Path) -> JsonObject:
        records = _jsonl_objects(path)
        session_id = _first_string(records, ["sessionId", "session_id", "uuid"]) or path.stem
        workspace = _first_string(records, ["cwd"])
        messages = [
            item for record in records for item in _claude_code_messages(record) if item["content"]
        ]
        return _conversation(
            ConversationRecord(
                provider=self.provider,
                source_path=path,
                source_conversation_id=session_id,
                title=None,
                created_at=_first_string(records, ["timestamp"]),
                updated_at=_last_string(records, ["timestamp"]),
                workspace=workspace,
                messages=messages,
            )
        )


class CodexAdapter:
    """Convert Codex rollout JSONL sessions into canonical chats."""

    provider = "codex"

    def conversations(self, source_dir: Path) -> list[JsonObject]:
        """Return canonical conversations from Codex session rollout files."""
        return [
            self._conversation_from_file(path)
            for path in sorted(source_dir.rglob("*.jsonl"))
            if path.is_file()
        ]

    def _conversation_from_file(self, path: Path) -> JsonObject:
        records = _jsonl_objects(path)
        meta = _codex_meta(records)
        messages = [
            message
            for record in records
            if record.get("type") == "response_item"
            for message in _codex_response_messages(record)
        ]
        return _conversation(
            ConversationRecord(
                provider=self.provider,
                source_path=path,
                source_conversation_id=meta.get("id") or path.stem,
                title=None,
                created_at=meta.get("created_at") or _first_string(records, ["timestamp"]),
                updated_at=_last_string(records, ["timestamp"]),
                workspace=meta.get("cwd"),
                messages=messages,
            )
        )


class OpenCodeAdapter:
    """Convert OpenCode file-backed storage into canonical chats."""

    provider = "opencode"

    def conversations(self, source_dir: Path) -> list[JsonObject]:
        """Return canonical conversations from OpenCode session/message/part files."""
        conversations: list[JsonObject] = []
        for session_path in sorted((source_dir / "session").rglob("ses_*.json")):
            session = _json_object(session_path)
            session_id = str(session.get("id", session_path.stem))
            messages = self._messages_for_session(source_dir, session_id)
            conversations.append(
                _conversation(
                    ConversationRecord(
                        provider=self.provider,
                        source_path=session_path,
                        source_conversation_id=session_id,
                        title=_optional_str(session.get("title")),
                        created_at=_nested_time(session, "created"),
                        updated_at=_nested_time(session, "updated"),
                        workspace=_optional_str(session.get("directory")),
                        messages=messages,
                    )
                )
            )
        return conversations

    def _messages_for_session(self, source_dir: Path, session_id: str) -> list[JsonObject]:
        messages: list[JsonObject] = []
        for message_path in sorted((source_dir / "message" / session_id).glob("msg_*.json")):
            message = _json_object(message_path)
            message_id = str(message.get("id", message_path.stem))
            messages.append(
                _message(
                    MessageRecord(
                        message_id=message_id,
                        parent_id=None,
                        role=_role(message.get("role")),
                        created_at=_nested_time(message, "created"),
                        content=self._parts_for_message(source_dir, message_id),
                        record_type="message",
                    )
                )
            )
        return messages

    def _parts_for_message(self, source_dir: Path, message_id: str) -> ContentItems:
        parts: ContentItems = []
        for part_path in sorted((source_dir / "part" / message_id).glob("prt_*.json")):
            part = _json_object(part_path)
            parts.extend(_opencode_part_content(part))
        return parts


def ensure_raw_chat_links(
    raw_chats_dir: Path | str = Path("data/raw/chats"),
    *,
    sources: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Create stable symlinks to raw chat source directories."""
    raw_dir = Path(raw_chats_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}
    for provider, source in (sources or DEFAULT_RAW_CHAT_SOURCES).items():
        source_path = source.expanduser()
        logger.info("raw_chat_source_check provider={} source={}", provider, source_path)
        if not source_path.exists():
            msg = f"Raw chat source for {provider} does not exist: {source_path}"
            raise FileNotFoundError(msg)
        link_path = raw_dir / provider
        if link_path.is_symlink():
            if link_path.resolve() != source_path.resolve():
                msg = f"Raw chat link for {provider} points to {link_path.resolve()}"
                raise FileExistsError(msg)
        elif link_path.exists():
            msg = f"Raw chat path for {provider} already exists and is not a symlink: {link_path}"
            raise FileExistsError(msg)
        else:
            link_path.symlink_to(source_path, target_is_directory=True)
        created[provider] = link_path
        logger.info("raw_chat_source_linked provider={} link={}", provider, link_path)
    return created


def write_canonical_chats(
    raw_chats_dir: Path | str = Path("data/raw/chats"),
    output_dir: Path | str = Path("data/canonical/chats"),
    *,
    providers: list[str] | None = None,
) -> dict[str, int]:
    """Write canonical JSONL chat files and return conversation counts."""
    raw_dir = Path(raw_chats_dir)
    canonical_dir = Path(output_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    selected = providers or list(_adapters())
    counts: dict[str, int] = {}
    for provider in selected:
        logger.info("canonical_chat_provider_start provider={}", provider)
        adapter = _adapter(provider)
        conversations = adapter.conversations(raw_dir / provider)
        output_path = canonical_dir / f"{provider}.jsonl"
        with output_path.open("w", encoding="utf-8") as output:
            for conversation in conversations:
                output.write(json.dumps(conversation, sort_keys=True) + "\n")
        counts[provider] = len(conversations)
        logger.info(
            "canonical_chat_provider_complete provider={} conversations={} output={}",
            provider,
            len(conversations),
            output_path,
        )
    return counts


def _adapters() -> dict[str, RawChatAdapter]:
    return {
        "claude-export": ClaudeExportAdapter(),
        "claude-code": ClaudeCodeAdapter(),
        "codex": CodexAdapter(),
        "opencode": OpenCodeAdapter(),
    }


def _adapter(provider: str) -> RawChatAdapter:
    adapters = _adapters()
    if provider not in adapters:
        allowed = ", ".join(adapters)
        msg = f"Unsupported raw chat provider '{provider}'. Allowed values: {allowed}"
        raise ValueError(msg)
    return adapters[provider]


def _conversation(record: ConversationRecord) -> JsonObject:
    return {
        "schema_version": 1,
        "provider": record.provider,
        "source_path": record.source_path.as_posix(),
        "source_conversation_id": record.source_conversation_id,
        "title": record.title,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "workspace": record.workspace,
        "messages": record.messages,
    }


def _message(record: MessageRecord) -> JsonObject:
    return {
        "id": record.message_id,
        "parent_id": record.parent_id,
        "role": record.role,
        "created_at": record.created_at,
        "content": record.content,
        "raw": {"provider_record_type": record.record_type},
    }


def _claude_message_content(message: JsonObject) -> ContentItems:
    content: ContentItems = []
    seen_text: set[str] = set()
    text = _optional_str(message.get("text"))
    if text:
        content.append({"type": "text", "text": text})
        seen_text.add(text)
    for item in _list_of_objects(message.get("content")):
        for content_item in _provider_content_item(item):
            item_text = _optional_str(content_item.get("text"))
            if content_item.get("type") == "text" and item_text in seen_text:
                continue
            if content_item.get("type") == "text" and item_text:
                seen_text.add(item_text)
            content.append(content_item)
    for attachment in _list_of_objects(message.get("attachments")):
        content.append(_attachment_content(attachment))
    for file_item in _list_of_objects(message.get("files")):
        content.append(_file_content(file_item))
    return content


def _claude_code_messages(record: JsonObject) -> list[JsonObject]:
    message = record.get("message")
    timestamp = _optional_str(record.get("timestamp"))
    if isinstance(message, dict):
        role = _role(message.get("role", record.get("type")))
        content = _message_content(message.get("content"))
        message_id = str(record.get("uuid", record.get("id", "")))
        parent_id = _optional_str(record.get("parentUuid"))
        return [
            _message(
                MessageRecord(
                    message_id=message_id,
                    parent_id=parent_id,
                    role=role,
                    created_at=timestamp,
                    content=content,
                    record_type=str(record.get("type", "message")),
                )
            )
        ]
    if record.get("type") in {"tool_use", "tool_result"}:
        return [_tool_event_message(record, timestamp)]
    return []


def _codex_response_messages(record: JsonObject) -> list[JsonObject]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    item_type = str(payload.get("type", ""))
    timestamp = _optional_str(record.get("timestamp"))
    if item_type == "message":
        return [_codex_message(payload, timestamp)]
    if item_type in {"function_call", "tool_call"}:
        return [_tool_call_message(payload, timestamp)]
    if item_type in {"function_call_output", "tool_result"}:
        return [_tool_result_message(payload, timestamp)]
    return []


def _codex_message(payload: JsonObject, timestamp: str | None) -> JsonObject:
    message_id = str(payload.get("id", payload.get("call_id", "")))
    return _message(
        MessageRecord(
            message_id=message_id,
            parent_id=_optional_str(payload.get("parent_id")),
            role=_role(payload.get("role")),
            created_at=timestamp,
            content=_message_content(payload.get("content")),
            record_type="message",
        )
    )


def _tool_event_message(record: JsonObject, timestamp: str | None) -> JsonObject:
    if record.get("type") == "tool_result":
        return _tool_result_message(record, timestamp)
    return _tool_call_message(record, timestamp)


def _tool_call_message(payload: JsonObject, timestamp: str | None) -> JsonObject:
    call_id = str(payload.get("call_id", payload.get("id", "")))
    content = {
        "type": "tool_call",
        "name": _optional_str(payload.get("name")),
        "status": "called",
    }
    return _message(
        MessageRecord(
            message_id=call_id,
            parent_id=_optional_str(payload.get("parent_id")),
            role="tool",
            created_at=timestamp,
            content=[content],
            record_type=str(payload.get("type", "tool_call")),
        )
    )


def _tool_result_message(payload: JsonObject, timestamp: str | None) -> JsonObject:
    call_id = str(payload.get("call_id", payload.get("id", "")))
    return _message(
        MessageRecord(
            message_id=f"{call_id}:result",
            parent_id=call_id or None,
            role="tool",
            created_at=timestamp,
            content=[{"type": "tool_result", "status": "completed"}],
            record_type=str(payload.get("type", "tool_result")),
        )
    )


def _message_content(value: object) -> ContentItems:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list):
        content: ContentItems = []
        for item in value:
            if isinstance(item, str):
                content.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                content.extend(_provider_content_item(cast("JsonObject", item)))
        return content
    return []


def _provider_content_item(item: JsonObject) -> ContentItems:
    item_type = str(item.get("type", ""))
    text = _optional_str(item.get("text"))
    if item_type in {"text", "input_text", "output_text"} and text:
        return [{"type": "text", "text": text}]
    if item_type in {"tool_use", "tool_call"}:
        return [{"type": "tool_call", "name": _optional_str(item.get("name")), "status": "called"}]
    if item_type in {"tool_result", "function_call_output"}:
        return [{"type": "tool_result", "status": "completed"}]
    if item_type == "reasoning":
        return [{"type": "reasoning"}]
    return []


def _opencode_part_content(part: JsonObject) -> ContentItems:
    part_type = str(part.get("type", ""))
    text = _optional_str(part.get("text"))
    if part_type == "text" and text:
        return _opencode_text_content(text)
    if "tool" in part_type:
        return _opencode_tool_content(part)
    return []


def _opencode_tool_content(part: JsonObject) -> ContentItems:
    content: ContentItems = [
        {"type": "tool_call", "name": _optional_str(part.get("tool")), "status": "called"}
    ]
    state = part.get("state")
    if isinstance(state, dict) and (
        state.get("output") is not None or state.get("status") == "completed"
    ):
        content.append({"type": "tool_result", "status": "completed"})
    return content


def _opencode_text_content(text: str) -> ContentItems:
    content: ContentItems = []
    text_lines: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip() == "<file>":
            _append_text_content(content, text_lines)
            index = _skip_opencode_file_block(lines, index)
            content.append({"type": "tool_result", "status": "completed"})
            continue
        tool_name = _opencode_tool_call_name(line)
        if tool_name is None:
            text_lines.append(line)
            index += 1
            continue
        _append_text_content(content, text_lines)
        content.append({"type": "tool_call", "name": tool_name, "status": "called"})
        index += 1
        if index < len(lines) and lines[index].strip() == "<file>":
            index = _skip_opencode_file_block(lines, index)
            content.append({"type": "tool_result", "status": "completed"})
    _append_text_content(content, text_lines)
    return content


def _append_text_content(content: ContentItems, lines: list[str]) -> None:
    text = "\n".join(lines).strip()
    lines.clear()
    if text:
        content.append({"type": "text", "text": text})


def _opencode_tool_call_name(line: str) -> str | None:
    prefix = "Called the "
    suffix = " tool with the following input:"
    stripped = line.strip()
    if not stripped.startswith(prefix):
        return None
    suffix_index = stripped.find(suffix, len(prefix))
    if suffix_index == -1:
        return None
    name = stripped[len(prefix) : suffix_index].strip()
    return name or "unknown"


def _skip_opencode_file_block(lines: list[str], start_index: int) -> int:
    index = start_index + 1
    while index < len(lines):
        if lines[index].strip() == "</file>":
            return index + 1
        index += 1
    return index


def _attachment_content(attachment: JsonObject) -> JsonObject:
    return {
        "type": "attachment",
        "id": _optional_str(attachment.get("id")),
        "name": _optional_str(attachment.get("file_name", attachment.get("name"))),
    }


def _file_content(file_item: JsonObject) -> JsonObject:
    return {
        "type": "file",
        "id": _optional_str(file_item.get("id")),
        "name": _optional_str(file_item.get("file_name", file_item.get("name"))),
    }


def _codex_meta(records: list[JsonObject]) -> dict[str, str | None]:
    for record in records:
        if record.get("type") != "session_meta":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        return {
            "id": _optional_str(payload.get("id", payload.get("session_id"))),
            "cwd": _optional_str(payload.get("cwd")),
            "created_at": _optional_str(payload.get("timestamp")),
        }
    return {"id": None, "cwd": None, "created_at": None}


def _json_array(path: Path) -> list[JsonObject]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Expected JSON array in {path}"
        raise TypeError(msg)
    return _list_of_objects(data)


def _json_object(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return data


def _jsonl_objects(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            msg = f"Expected JSON object in {path}:{line_number}"
            raise TypeError(msg)
        records.append(data)
    return records


def _list_of_objects(value: object) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [cast("JsonObject", item) for item in value if isinstance(item, dict)]


def _first_string(records: list[JsonObject], keys: list[str]) -> str | None:
    for record in records:
        for key in keys:
            value = _optional_str(record.get(key))
            if value:
                return value
    return None


def _last_string(records: list[JsonObject], keys: list[str]) -> str | None:
    for record in reversed(records):
        for key in keys:
            value = _optional_str(record.get(key))
            if value:
                return value
    return None


def _nested_time(data: JsonObject, key: str) -> str | None:
    time_value = data.get("time")
    if isinstance(time_value, dict):
        return _optional_str(time_value.get(key))
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _claude_role(value: object) -> str:
    role = str(value).lower()
    if role in {"human", "user"}:
        return "user"
    return _role(value)


def _role(value: object) -> str:
    role = str(value).lower()
    if role in {"user", "assistant", "system", "tool"}:
        return role
    return "unknown"
