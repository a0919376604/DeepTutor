"""Unit tests for shared CLI helpers."""

from __future__ import annotations

import pytest

from deeptutor_cli.common import parse_json_object


def test_parse_json_object_whitespace_is_empty_dict() -> None:
    assert parse_json_object("   ") == {}


def test_parse_json_object_invalid_json_still_errors() -> None:
    with pytest.raises(ValueError, match="Invalid JSON config"):
        parse_json_object("{")


_CATALOG = {
    "services": {
        "llm": {
            "active_profile_id": "p-cc",
            "active_model_id": "m-sonnet",
            "profiles": [
                {
                    "id": "p-cc",
                    "models": [
                        {"id": "m-sonnet", "name": "sonnet", "model": "sonnet"},
                        {"id": "m-opus", "name": "Opus", "model": "opus"},
                    ],
                },
                {"id": "p-other", "models": [{"id": "m-gpt", "name": "gpt", "model": "gpt-5"}]},
            ],
        }
    }
}


def test_resolve_llm_selection_matches_model_name_in_active_profile_first() -> None:
    from deeptutor_cli.common import resolve_llm_selection

    assert resolve_llm_selection("OPUS", _CATALOG) == {"profile_id": "p-cc", "model_id": "m-opus"}
    assert resolve_llm_selection("gpt-5", _CATALOG) == {"profile_id": "p-other", "model_id": "m-gpt"}
    assert resolve_llm_selection(None, _CATALOG) is None


def test_resolve_llm_selection_unknown_model_lists_available_names() -> None:
    from deeptutor_cli.common import resolve_llm_selection

    with pytest.raises(ValueError, match="sonnet, opus, gpt-5"):
        resolve_llm_selection("haiku", _CATALOG)


def test_build_turn_request_carries_llm_selection() -> None:
    from deeptutor_cli.common import build_turn_request

    request = build_turn_request(
        content="hi", capability="chat", session_id=None, tools=[], knowledge_bases=[],
        language="en", config_items=[], config_json=None, notebook_refs=[], history_refs=[],
        llm_selection={"profile_id": "p-cc", "model_id": "m-opus"},
    )
    assert request.llm_selection is not None
    assert request.llm_selection.model_id == "m-opus"


def test_resolve_llm_selection_effort_only_uses_active_model() -> None:
    from deeptutor_cli.common import resolve_llm_selection

    assert resolve_llm_selection(None, _CATALOG, effort="HIGH") == {
        "profile_id": "p-cc",
        "model_id": "m-sonnet",
        "reasoning_effort": "high",
    }
    assert resolve_llm_selection("opus", _CATALOG, effort="max")["reasoning_effort"] == "max"


def test_resolve_llm_selection_rejects_unknown_effort() -> None:
    from deeptutor_cli.common import resolve_llm_selection

    with pytest.raises(ValueError, match="reasoning_effort"):
        resolve_llm_selection(None, _CATALOG, effort="turbo")


def test_turn_request_selection_carries_reasoning_effort() -> None:
    from deeptutor_cli.common import build_turn_request

    request = build_turn_request(
        content="hi", capability="chat", session_id=None, tools=[], knowledge_bases=[],
        language="en", config_items=[], config_json=None, notebook_refs=[], history_refs=[],
        llm_selection={"profile_id": "p-cc", "model_id": "m-opus", "reasoning_effort": "high"},
    )
    assert request.llm_selection.reasoning_effort == "high"
