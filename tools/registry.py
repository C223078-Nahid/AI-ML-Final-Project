"""Centralized tool registry (follow-up task 4, sections 6.2, 6.3, 11).

Every tool the assistant can ever invoke is registered here exactly once,
with:

- a meaningful name,
- a clear description (the model sees it),
- a validated input schema (Pydantic ``args_schema`` - every argument is
  validated independently of the model),
- a permission boundary:

    READ_ONLY  - safe to bind to the model (search, read, list, verify).
    RESTRICTED - application-invoked ONLY (e.g. create_task after human
                 approval).  A restricted tool is NEVER handed to a model
                 through ``bind_tools``; the model cannot even request it,
                 and the model's decision to call a tool never grants
                 permission to execute a restricted operation.

The registry is deliberately framework-agnostic: it stores LangChain tools
(anything with ``.name`` / ``.invoke`` and an ``args_schema``), so the
graph nodes and the agent loop share ONE source of truth for what exists,
what is available, and what is safe.

Not exposed as tools (by design): database connections, arbitrary SQL,
shell execution, unrestricted filesystem access, arbitrary HTTP fetching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ToolPermission(str, Enum):
    READ_ONLY = "read_only"
    RESTRICTED = "restricted"


@dataclass
class RegisteredTool:
    """One registered tool and its permission boundary."""

    name: str
    description: str
    permission: ToolPermission
    tool: Any
    args_schema: type[BaseModel] | None = None
    #: False when the tool exists but is not usable in this configuration
    #: (e.g. web_search without an API key).  Unavailable tools are NOT
    #: bound to the model; the Settings page reports them.
    available: bool = True
    unavailable_reason: str = ""
    #: Optional non-sensitive observability metadata (e.g. which search
    #: provider serves the web_search tool - follow-up task 4, section 13).
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}

    @property
    def bindable(self) -> bool:
        """Whether this tool may be handed to a language model."""
        return self.permission is ToolPermission.READ_ONLY and self.available

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description[:200],
            "permission": self.permission.value,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "bindable": self.bindable,
        }


class ToolRegistry:
    """The single registry of application tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    # -- registration -------------------------------------------------------------
    def register(
        self,
        tool: Any,
        *,
        permission: ToolPermission,
        description: str | None = None,
        available: bool = True,
        unavailable_reason: str = "",
        meta: dict | None = None,
    ) -> RegisteredTool:
        """Register a tool with its permission boundary.

        Validates the tool shape (unique name, non-empty description,
        optional Pydantic args_schema) so a misconfigured tool fails loudly
        at registration time - never silently at model time.
        """
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        if not name or not isinstance(name, str):
            raise ValueError("a registered tool must have a non-empty name")
        if name in self._tools:
            raise ValueError(f"a tool named '{name}' is already registered")
        desc = description or (getattr(tool, "description", "") or "").strip()
        if not desc:
            raise ValueError(f"tool '{name}' must have a clear description")
        schema = getattr(tool, "args_schema", None)
        if schema is not None and not (
            isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            raise ValueError(
                f"tool '{name}' args_schema must be a Pydantic model"
            )
        if permission not in ToolPermission:
            raise ValueError(f"unknown permission: {permission!r}")
        entry = RegisteredTool(
            name=name,
            description=desc,
            permission=permission,
            tool=tool,
            args_schema=schema,
            available=available,
            unavailable_reason=unavailable_reason,
            meta=meta or {},
        )
        self._tools[name] = entry
        return entry

    # -- lookup ---------------------------------------------------------------------
    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def all_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def bindable_tools(self) -> list[Any]:
        """The READ-ONLY, available tools - the only set ever bound to a
        model via ``gateway.chat(..., tools=...)`` / ``bind_tools``."""
        return [
            entry.tool
            for entry in self._tools.values()
            if entry.bindable
        ]

    def describe(self) -> list[dict[str, Any]]:
        """Non-sensitive registry snapshot for the Settings page/tests."""
        return [entry.describe() for entry in self._tools.values()]


# ---------------------------------------------------------------------------
# Default application registry
# ---------------------------------------------------------------------------
def build_default_registry(
    *,
    retriever: Any,
    doc_repo: Any,
    task_repo: Any,
    settings: Any,
    search_service: Any = None,
    collector: dict | None = None,
) -> ToolRegistry:
    """Register every application tool (follow-up task 4, section 6.2).

    Used by ``app.build_container`` (one registry per application) and as
    the lazy fallback in ``graph.nodes.make_nodes`` for direct test
    construction.  ``collector`` optionally receives every chunk the model
    retrieved through search_documents in this run (used for citation
    validation); the graph passes a fresh dict per request.

    Registry contents (name / permission):

    - search_documents        read-only   - search the shared library
    - get_document_content    read-only   - fetch a document's passages
    - list_tasks              read-only   - list saved tasks
    - get_current_datetime    read-only   - application clock (timezone-aware)
    - web_search              read-only   - optional; only when configured
    - read_web_page           read-only   - restricted page extraction
                                           (search-result URLs only)
    - verify_claims           read-only   - deterministic claim/evidence
                                           comparison used by fact-checking
    - create_task             RESTRICTED  - never bound to a model; invoked
                                           only by task_execution after
                                           human approval
    """
    from core.web_search import WebSearchService, build_search_service
    from tools.datetime_tools import make_datetime_tool
    from tools.document_tools import (
        make_get_document_content_tool,
        make_search_documents_tool,
    )
    from tools.task_tools import make_create_task_tool, make_list_tasks_tool
    from tools.verify_claims_tool import make_verify_claims_tool
    from tools.web_tools import (
        make_read_web_page_tool,
        make_web_search_tool,
    )

    registry = ToolRegistry()
    run_collector: dict = collector if collector is not None else {}

    registry.register(
        make_search_documents_tool(
            retriever, settings.retrieval_top_k, run_collector
        ),
        permission=ToolPermission.READ_ONLY,
    )
    registry.register(
        make_get_document_content_tool(
            doc_repo, retriever, max_chunks=8
        ),
        permission=ToolPermission.READ_ONLY,
    )
    registry.register(
        make_list_tasks_tool(task_repo),
        permission=ToolPermission.READ_ONLY,
    )
    registry.register(
        make_datetime_tool(),
        permission=ToolPermission.READ_ONLY,
    )
    registry.register(
        make_verify_claims_tool(),
        permission=ToolPermission.READ_ONLY,
    )
    registry.register(
        make_create_task_tool(task_repo),
        permission=ToolPermission.RESTRICTED,
        description=(
            "RESTRICTED WRITE TOOL: create a task. Never bound to a model; "
            "invoked only by the application's task_execution node after "
            "the user explicitly approved the proposal."
        ),
    )

    service = search_service
    if not isinstance(service, WebSearchService):
        service = build_search_service(settings)
    web_available = service is not None and service.available
    registry.register(
        make_web_search_tool(service),
        permission=ToolPermission.READ_ONLY,
        available=web_available,
        unavailable_reason=(
            ""
            if web_available
            else "web search disabled or TAVILY_API_KEY not configured"
        ),
        meta={"search_provider": service.provider_name},
    )
    registry.register(
        make_read_web_page_tool(service),
        permission=ToolPermission.READ_ONLY,
        available=web_available,
        unavailable_reason=(
            ""
            if web_available
            else "web search disabled or TAVILY_API_KEY not configured"
        ),
    )
    return registry
