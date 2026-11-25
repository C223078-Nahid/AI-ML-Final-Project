"""Follow-up task 4, section 6.2: the centralized tool registry.

Contracts:
- every tool registers with a name, description, validated schema, and a
  permission boundary;
- ONLY read-only available tools are bindable to a model;
- the restricted create_task tool can never be bound;
- unconfigured web search is reported unavailable and excluded;
- misconfigured registrations fail loudly.
"""

from __future__ import annotations

import pytest

from tools.registry import (
    ToolPermission,
    ToolRegistry,
    build_default_registry,
)


def _build(services_settings=None, **overrides):
    from config.settings import Settings

    settings = services_settings or Settings(
        _env_file=None,
        data_dir="/tmp/pkta-registry",
        qdrant_path="/tmp/pkta-registry/q",
        sqlite_path="/tmp/pkta-registry/a.db",
        checkpoint_db_path="/tmp/pkta-registry/c.db",
    )
    kwargs = dict(
        retriever=None,
        doc_repo=None,
        task_repo=None,
        settings=settings,
    )
    kwargs.update(overrides)
    return build_default_registry(**kwargs)


def test_registry_contains_the_required_tools():
    """The exact tool set from the task's registry table."""
    registry = _build()
    names = {entry.name for entry in registry.all_tools()}
    assert {
        "search_documents",
        "get_document_content",
        "list_tasks",
        "get_current_datetime",
        "web_search",
        "verify_claims",
        "create_task",
    } <= names


def test_create_task_is_restricted_and_never_bindable():
    registry = _build()
    entry = registry.get("create_task")
    assert entry.permission is ToolPermission.RESTRICTED
    assert entry.bindable is False
    bindable_names = {
        getattr(t, "name", str(t)) for t in registry.bindable_tools()
    }
    assert "create_task" not in bindable_names


def test_read_only_tools_are_bindable():
    registry = _build()
    for name in (
        "search_documents",
        "get_document_content",
        "list_tasks",
        "get_current_datetime",
        "verify_claims",
    ):
        entry = registry.get(name)
        assert entry.permission is ToolPermission.READ_ONLY
        assert entry.bindable is True


def test_web_search_unavailable_without_key():
    """Section 9: gracefully unavailable when the API key is missing."""
    registry = _build()
    entry = registry.get("web_search")
    assert entry.available is False
    assert entry.bindable is False
    assert entry.unavailable_reason
    assert "read_web_page" not in {
        getattr(t, "name", str(t)) for t in registry.bindable_tools()
    }


def test_web_search_available_with_key():
    from core.web_search import WebSearchService
    from tests.test_web_search import FakeTavily

    service = WebSearchService(FakeTavily())
    registry = _build(search_service=service)
    entry = registry.get("web_search")
    assert entry.available is True
    assert entry.bindable is True
    assert "web_search" in {
        getattr(t, "name", str(t)) for t in registry.bindable_tools()
    }


def test_registration_validates_tool_shape():
    registry = ToolRegistry()

    class NoName:
        pass

    with pytest.raises(ValueError, match="non-empty name"):
        registry.register(NoName(), permission=ToolPermission.READ_ONLY)

    from tools.task_tools import make_list_tasks_tool

    list_tool = make_list_tasks_tool(None)
    registry.register(list_tool, permission=ToolPermission.READ_ONLY)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(list_tool, permission=ToolPermission.READ_ONLY)


def test_every_bindable_tool_has_a_pydantic_schema():
    """Every model-facing tool validates its arguments with a schema."""
    registry = _build()
    for tool in registry.bindable_tools():
        schema = getattr(tool, "args_schema", None)
        assert schema is not None, f"{tool.name} lacks an args_schema"
        from pydantic import BaseModel

        assert isinstance(schema, type) and issubclass(schema, BaseModel)


def test_registry_describe_is_non_sensitive():
    registry = _build()
    for entry in registry.describe():
        assert set(entry) == {
            "name",
            "description",
            "permission",
            "available",
            "unavailable_reason",
            "bindable",
        }
        assert "key" not in entry["description"].lower()
        assert "token" not in entry["description"].lower()
