from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Source(StrEnum):
    """Known v0 memory sources."""

    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    OPENCODE = "opencode"
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"

    @classmethod
    def parse(cls, value: Source | str) -> Source:
        """Parse a source string into the constrained v0 enum."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            msg = f"Unsupported source '{value}'. Allowed values: {allowed}"
            raise ValueError(msg) from error


class MemoryType(StrEnum):
    """Known v0 memory classifications."""

    PREFERENCE = "preference"
    FACT = "fact"
    PROJECT_CONTEXT = "project_context"
    DECISION = "decision"
    PATTERN = "pattern"
    WARNING = "warning"

    @classmethod
    def parse(cls, value: MemoryType | str) -> MemoryType:
        """Parse a memory-type string into the constrained v0 enum."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            msg = f"Unsupported memory_type '{value}'. Allowed values: {allowed}"
            raise ValueError(msg) from error


def validate_importance(value: float) -> float:
    """Validate that an importance score is inside the v0 range."""
    if 0.0 <= value <= 1.0:
        return value
    msg = "Importance must be between 0.0 and 1.0"
    raise ValueError(msg)


@dataclass(frozen=True)
class AddMemory:
    """Input for creating a memory."""

    content: str
    source: Source | str = Source.CODEX
    memory_type: MemoryType | str = MemoryType.FACT
    importance: float = 0.5

    def __post_init__(self) -> None:
        """Normalize and validate creation input."""
        content = self.content.strip()
        if not content:
            msg = "Memory content cannot be empty"
            raise ValueError(msg)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "source", Source.parse(self.source))
        object.__setattr__(self, "memory_type", MemoryType.parse(self.memory_type))
        object.__setattr__(self, "importance", validate_importance(self.importance))


@dataclass(frozen=True)
class Memory:
    """Persisted memory record."""

    id: str
    content: str
    source: Source
    memory_type: MemoryType
    importance: float
    created_at: int
    updated_at: int
