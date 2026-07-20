from dataclasses import FrozenInstanceError

import pytest

from llm_api_resilience import PromptProfile


pytestmark = pytest.mark.unit


def test_empty_profile_is_a_no_op():
    profile = PromptProfile()

    assert profile.is_empty is True
    assert profile.to_messages() == ()
    assert profile.apply_to(({"role": "user", "content": "Hello"},)) == (
        {"role": "user", "content": "Hello"},
    )


def test_profile_normalizes_system_and_developer_instructions():
    profile = PromptProfile(system="Be concise.", developer="Use plain language.")

    assert profile.is_empty is False
    assert profile.to_messages() == (
        {"role": "system", "content": "Be concise."},
        {"role": "developer", "content": "Use plain language."},
    )


def test_apply_to_prepends_profile_and_does_not_mutate_messages():
    messages = [{"role": "user", "content": {"text": "Hello"}}]
    profile = PromptProfile(system="Be helpful.")

    result = profile.apply_to(messages)
    result[1]["content"]["text"] = "Changed"

    assert messages == [{"role": "user", "content": {"text": "Hello"}}]
    assert result == (
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": {"text": "Changed"}},
    )


def test_profile_is_immutable_and_does_not_leak_instruction_text_in_repr():
    profile = PromptProfile(system="private system instruction")

    with pytest.raises(FrozenInstanceError):
        profile.system = "changed"

    assert "private system instruction" not in repr(profile)
    assert repr(profile) == "PromptProfile(has_system=True, has_developer=False)"


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    [
        ("system", 123, TypeError),
        ("developer", object(), TypeError),
        ("system", "   ", ValueError),
        ("developer", "", ValueError),
    ],
)
def test_profile_validates_instruction_values(field_name, value, error):
    with pytest.raises(error):
        PromptProfile(**{field_name: value})


def test_apply_to_rejects_a_single_message_instead_of_an_iterable():
    with pytest.raises(TypeError, match="iterable of message mappings"):
        PromptProfile(system="Be helpful.").apply_to(
            {"role": "user", "content": "Hello"}
        )
