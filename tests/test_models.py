import pytest

from memories.models import MemoryType, Source, validate_importance


@pytest.mark.parametrize(
    "source",
    ["chatgpt", "claude", "opencode", "codex", "claude-code"],
)
def test_source_accepts_v0_enum_values(source: str) -> None:
    assert Source.parse(source).value == source


def test_source_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported source"):
        Source.parse("cli")


@pytest.mark.parametrize(
    "memory_type",
    ["preference", "fact", "project_context", "decision", "pattern", "warning"],
)
def test_memory_type_accepts_v0_enum_values(memory_type: str) -> None:
    assert MemoryType.parse(memory_type).value == memory_type


def test_memory_type_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported memory_type"):
        MemoryType.parse("note")


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_importance_accepts_zero_to_one(value: float) -> None:
    assert validate_importance(value) == value


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_importance_rejects_values_outside_zero_to_one(value: float) -> None:
    with pytest.raises(ValueError, match=r"Importance must be between 0\.0 and 1\.0"):
        validate_importance(value)
