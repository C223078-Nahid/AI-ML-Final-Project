"""Follow-up task 4, section 9: the web-search service and its tools.

All tests run offline against a scripted Tavily provider (no API calls).
Contracts:
- provider-independent service architecture (one provider, extensible);
- results carry title / URL / domain / snippet / dates / retrieval stamp;
- results are wrapped as UNTRUSTED external data (prompt-injection guard);
- the restricted page-extraction tool accepts ONLY URLs returned by
  web_search in the same run, with full SSRF protections;
- an unconfigured service is gracefully unavailable (never a crash).
"""

from __future__ import annotations

import json

import pytest

from core.web_search import (
    SearchProvider,
    SearchResult,
    WebSearchError,
    WebSearchService,
    WebSearchUnavailable,
    build_search_service,
    format_search_results,
    _is_public_host,
)
from tools.web_tools import make_web_search_tool


class FakeTavily(SearchProvider):
    """Scripted provider: returns fixed results, records the query."""

    name = "tavily"

    def __init__(self, results=None, error: Exception | None = None):
        self.results = results if results is not None else [
            SearchResult(
                title="Gold - Element information (Royal Society of Chemistry)",
                url="https://www.rsc.org/periodic-table/element/79/gold",
                domain="rsc.org",
                snippet="Gold is a chemical element with symbol Au.",
                publication_date="2024-01-01",
            ),
        ]
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str, max_results: int):
        if self.error:
            raise self.error
        self.queries.append(query)
        return self.results[:max_results]


def _service() -> WebSearchService:
    return WebSearchService(FakeTavily())


# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------
def test_unconfigured_service_is_unavailable():
    from config.settings import Settings

    settings = Settings(_env_file=None, web_search_enabled=True)
    service = build_search_service(settings)
    assert service.available is False
    with pytest.raises(WebSearchUnavailable):
        service.search("gold symbol")


def test_disabled_service_is_unavailable():
    from config.settings import Settings

    settings = Settings(
        _env_file=None, web_search_enabled=False, tavily_api_key="tvly-x"
    )
    assert build_search_service(settings).available is False


def test_configured_service_uses_tavily():
    from config.settings import Settings

    settings = Settings(_env_file=None, tavily_api_key="tvly-test")
    service = build_search_service(settings)
    assert service.available is True
    assert service.provider_name == "tavily"


# ---------------------------------------------------------------------------
# Search behavior
# ---------------------------------------------------------------------------
def test_search_returns_structured_results_with_metadata():
    service = _service()
    service.reset_run_scope()
    results = service.search("chemical symbol gold")
    assert results and results[0].url.startswith("https://")
    assert results[0].domain == "rsc.org"
    assert results[0].publication_date == "2024-01-01"
    assert results[0].retrieved_at  # retrieval timestamp always present
    as_dict = results[0].to_dict()
    assert set(as_dict) == {
        "title", "url", "domain", "snippet", "publication_date",
        "retrieved_at",
    }


def test_search_validates_the_query():
    service = _service()
    with pytest.raises(WebSearchError):
        service.search("   ")


def test_provider_failure_is_a_clear_error_not_a_crash():
    service = WebSearchService(
        FakeTavily(error=WebSearchError("search provider returned HTTP 500"))
    )
    with pytest.raises(WebSearchError, match="HTTP 500"):
        service.search("anything")


def test_results_are_wrapped_as_untrusted():
    """Prompt-injection protection: external data is delimited."""
    service = _service()
    service.reset_run_scope()
    results = service.search("gold")
    rendered = format_search_results(results)
    from core.guardrails import UNTRUSTED_BEGIN, UNTRUSTED_END

    assert UNTRUSTED_BEGIN in rendered and UNTRUSTED_END in rendered
    assert "untrusted" in rendered.lower()
    assert "rsc.org" in rendered  # real citation metadata travels with it


# ---------------------------------------------------------------------------
# Restricted page extraction (SSRF + allowlist)
# ---------------------------------------------------------------------------
def test_read_page_rejects_urls_not_returned_by_search():
    service = _service()
    service.reset_run_scope()
    with pytest.raises(WebSearchError, match="not returned by web_search"):
        service.read_page("https://en.wikipedia.org/wiki/Gold")


def test_read_page_allowlist_is_run_scoped():
    """URLs remembered in one run are forgotten in the next."""
    service = _service()
    service.reset_run_scope()
    service.search("gold")  # remembers rsc.org/...
    service.reset_run_scope()  # new request
    with pytest.raises(WebSearchError):
        service.read_page("https://www.rsc.org/periodic-table/element/79/gold")


class _ScriptedFetchService(WebSearchService):
    """Service whose read_page fetch is scripted at the httpx layer."""

    def __init__(self, provider, handler):
        super().__init__(provider)
        self._handler = handler

    def read_page(self, url):
        import core.web_search as ws

        real_httpx_get = ws.httpx_get if hasattr(ws, "httpx_get") else None
        # call the parent logic with a patched httpx module
        import httpx

        class _Resp:
            status_code = 200
            text = "<html><body><p>Gold has the symbol Au.</p></body></html>"
            url = url
            headers = {"content-type": "text/html"}

        class _FakeHttpx:
            TimeoutException = httpx.TimeoutException
            HTTPError = httpx.HTTPError

            @staticmethod
            def get(target, timeout=None, follow_redirects=False, headers=None):
                return self._handler(target) or _Resp()

        import types

        fake_module = types.ModuleType("httpx")
        fake_module.TimeoutException = httpx.TimeoutException
        fake_module.HTTPError = httpx.HTTPError
        fake_module.get = _FakeHttpx.get

        import builtins

        original_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "httpx":
                return fake_module
            return original_import(name, *args, **kwargs)

        builtins.__import__ = _patched_import
        try:
            return super().read_page(url)
        finally:
            builtins.__import__ = original_import


def test_read_page_extracts_text_for_allowlisted_url(monkeypatch):
    import httpx

    service = _service()
    service.reset_run_scope()
    service.search("gold")  # allowlists the rsc.org URL

    class _Resp:
        status_code = 200
        url = "https://www.rsc.org/periodic-table/element/79/gold"
        text = (
            "<html><head><style>body{}</style></head>"
            "<body><script>evil()</script>"
            "<p>Gold is a chemical element with the symbol Au.</p></body></html>"
        )
        headers = {"content-type": "text/html; charset=utf-8"}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    text = service.read_page(
        "https://www.rsc.org/periodic-table/element/79/gold"
    )
    assert "symbol Au" in text
    assert "evil()" not in text  # scripts stripped


def test_read_page_blocks_private_and_local_hosts():
    assert _is_public_host("localhost") is False
    assert _is_public_host("127.0.0.1") is False
    assert _is_public_host("10.1.2.3") is False
    assert _is_public_host("192.168.1.10") is False
    assert _is_public_host("172.16.0.5") is False
    assert _is_public_host("169.254.169.254") is False  # cloud metadata
    assert _is_public_host("[::1]") is False
    assert _is_public_host("example.com") is True


# ---------------------------------------------------------------------------
# Model-facing tool
# ---------------------------------------------------------------------------
def test_web_search_tool_returns_untrusted_evidence():
    service = _service()
    tool = make_web_search_tool(service)
    service.reset_run_scope()
    output = tool.invoke({"query": "chemical symbol of gold", "max_results": 3})
    assert "rsc.org" in output
    assert "UNTRUSTED" in output


def test_web_search_tool_reports_unavailability_clearly():
    from config.settings import Settings

    service = build_search_service(Settings(_env_file=None))
    tool = make_web_search_tool(service)
    output = tool.invoke({"query": "anything at all", "max_results": 3})
    assert output.startswith("Web search is unavailable")
    assert "TAVILY_API_KEY" in output


def test_web_search_tool_reports_provider_errors():
    service = WebSearchService(
        FakeTavily(error=WebSearchError("search provider rate limit"))
    )
    tool = make_web_search_tool(service)
    output = tool.invoke({"query": "anything", "max_results": 3})
    assert output.startswith("Web search failed")
    # never a crash: the model receives a usable error string.


def test_web_search_tool_validates_arguments():
    """Arguments are validated independently of the model (section 11)."""
    service = _service()
    tool = make_web_search_tool(service)
    with pytest.raises(Exception):
        tool.invoke({"query": "ab", "max_results": 3})  # too short
    with pytest.raises(Exception):
        tool.invoke({"query": "a valid query", "max_results": 99})  # > 10
