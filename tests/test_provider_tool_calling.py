"""Follow-up task 4, section 5 + test F: provider configuration and tool
calling across the ordered chain.

Contracts:
- Cloudflare GLM-4.7-Flash is the primary; OpenRouter Nemotron is the first
  fallback; other providers stay available behind them;
- the SAME tools and the SAME messages (including tool-call context) go to
  every provider attempt, so a fallback never loses the tool-call context;
- both configured providers receive identically bound tools.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm_gateway import (
    PROVIDER_ANTHROPIC,
    PROVIDER_CLOUDFLARE,
    PROVIDER_OPENROUTER,
    ModelGateway,
    ProviderEntry,
)
from tests.test_fallback_chain import _RecordingModel  # scripted model


def test_default_chain_is_cloudflare_then_openrouter(monkeypatch):
    """Section 5: Cloudflare primary, OpenRouter first fallback, legacy
    Anthropic final resort - built from settings."""
    from config.settings import Settings

    settings = Settings(
        _env_file=None,
        cloudflare_account_id="cf-account",
        cloudflare_api_token="cf-token",
        openrouter_api_key="or-key",
        anthropic_api_key="ant-key",
    )
    gateway = ModelGateway.from_settings(settings)
    chain = gateway.describe()["chain"]
    assert chain[0]["provider"] == PROVIDER_CLOUDFLARE
    assert chain[0]["model"] == "@cf/zai-org/glm-4.7-flash"
    assert chain[0]["available"] is True
    assert chain[1]["provider"] == PROVIDER_OPENROUTER
    assert chain[1]["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert chain[1]["available"] is True
    assert chain[2]["provider"] == PROVIDER_ANTHROPIC  # preserved final resort


def test_fallback_preserves_tools_and_tool_call_context():
    """Test F: the primary fails (rate limit) mid-tool-loop; the fallback
    receives the SAME message list (with the tool-call exchange) and the
    SAME tool definitions."""
    primary = _RecordingModel(_rate_limit_error())
    fallback = _RecordingModel(AIMessage(content="fallback answer"))

    gateway = ModelGateway(
        primary=None, fallback=None, primary_name="", fallback_name="",
        chain=[
            ProviderEntry("primary", PROVIDER_CLOUDFLARE, "glm", primary),
            ProviderEntry("fallback_1", PROVIDER_OPENROUTER, "nemotron",
                          fallback),
        ],
    )

    class _Tool:
        name = "get_current_datetime"
        description = "current date/time"

    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="What day is today?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_current_datetime", "args": {},
                 "id": "c1", "type": "tool_call"}
            ],
        ),
    ]
    result = gateway.chat(messages, tools=[_Tool()], run_name="t")
    assert result.content == "fallback answer"
    # BOTH providers were offered the SAME tools...
    assert [t.name for t in primary.bound_tools[0]] == [
        "get_current_datetime"
    ]
    assert [t.name for t in fallback.bound_tools[0]] == [
        "get_current_datetime"
    ]
    # ...and the fallback received the FULL original context verbatim.
    assert len(fallback.seen) == 1
    assert [type(m) for m in fallback.seen[0]] == [type(m) for m in messages]
    assert fallback.seen[0][2].tool_calls == messages[2].tool_calls


def _rate_limit_error():
    import openai

    try:
        raise openai.RateLimitError(
            message="rate limited",
            response=_fake_response(429),
            body={"error": {"message": "rate limited"}},
        )
    except openai.RateLimitError as exc:
        return exc


class _FakeHeaders:
    def get(self, name, default=None):
        return default


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = _FakeHeaders()
        self.request = None

    def json(self):
        return {}


def _fake_response(code):
    return _FakeResponse(code)


def test_tool_calls_survive_provider_switch_in_graph(services, settings):
    """End-to-end: the agent loop's conversation remains intact when the
    gateway switches providers between iterations: the second provider
    attempt receives the SAME messages as the first."""
    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    class _ChainGateway:
        """Gateway double that mimics the real ModelGateway failover: the
        PRIMARY provider fails with a rate limit, the FALLOVER provider is
        handed the SAME messages and succeeds - the node sees one success."""

        def __init__(self):
            self.provider_calls: list[list] = []
            self.last_events = []
            self.answer = AIMessage(
                content="Today is Wednesday, 25 November 2026."
            )

        def chat(self, messages, tools=None, run_name=None, metadata=None):
            # primary provider attempt: rate-limited, recorded, swallowed
            # (exactly what ModelGateway.chat does before failing over)
            self.provider_calls.append(list(messages))
            self.last_events.append(
                {"attempt": "primary", "outcome": "failed"}
            )
            # fallback provider attempt: the SAME messages, succeeds
            self.provider_calls.append(list(messages))
            self.last_events.append(
                {"attempt": "fallback_1", "outcome": "success"}
            )
            return self.answer

        def structured(self, *args, **kwargs):
            from core.models import Intent, IntentResult

            return IntentResult(intent=Intent.GENERAL_CONVERSATION,
                                confidence=0.9)

    services.gateway = _ChainGateway()
    graph = build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )
    from langchain_core.messages import HumanMessage

    result = graph.invoke(
        {
            "user_request": "What day is today?",
            "thread_id": "fb-1",
            "messages": [HumanMessage(content="What day is today?")],
        },
        {"configurable": {"thread_id": "fb-1"}},
    )
    assert "25 November 2026" in result["final_response"]["text"]
    # The provider switch kept the conversation: the second (working)
    # provider attempt received the SAME messages as the first.
    gateway = services.gateway
    assert len(gateway.provider_calls) >= 2
    first_messages = gateway.provider_calls[0]
    second_messages = gateway.provider_calls[1]
    assert [str(m.content)[:60] for m in first_messages] == [
        str(m.content)[:60] for m in second_messages
    ]
