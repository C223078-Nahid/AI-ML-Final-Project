"""Ordered provider failover tests (follow-up task, sections 6.5-6.7).

Simulates, with scripted models and REAL openai/anthropic SDK exceptions:

1. OpenRouter success (no fallback).
2. OpenRouter HTTP 429            -> Cloudflare success.
3. OpenRouter quota exhaustion    -> Cloudflare success.
4. OpenRouter timeout             -> Cloudflare success.
5. Cloudflare failure             -> legacy Anthropic fallback success.
6. All providers unavailable      -> descriptive ModelGatewayError.
7. Authentication errors on the   -> eligible, chain continues.
  primary
8. Invalid requests (bad model    -> NOT eligible: clear config error,
  identifiers)                        no uncontrolled retries.

Also verifies the execution-consistency requirements: the same message
list (prompt + conversation context + retrieved RAG passages) and the
same tool definitions reach every attempt, and LangSmith metadata records
the provider, model, duration, and fallback reason.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

import anthropic
import openai

from core.llm_gateway import (
    PROVIDER_ANTHROPIC,
    PROVIDER_CLOUDFLARE,
    PROVIDER_OPENROUTER,
    ErrorCategory,
    ModelGateway,
    ModelGatewayError,
    ProviderEntry,
)
from tests.fakes import (
    ScriptedModel,
    make_anthropic_error,
    make_openai_error,
)

MESSAGES = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Summarize the Qdrant section of my notes."),
]
RAG_PASSAGE = (
    "S1 [notes.txt] The vector database is Qdrant; it stores the "
    "document embeddings locally."
)
RAG_MESSAGES = MESSAGES + [HumanMessage(content=RAG_PASSAGE)]

NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"
GLM = "@cf/zai-org/glm-4.7-flash"
HAIKU = "claude-haiku-5-5"


class _RecordingModel(ScriptedModel):
    """ScriptedModel that records every received message list and config."""

    def __init__(self, behavior, structured_result=None):
        super().__init__(behavior, structured_result)
        self.seen: list[list] = []
        self.configs: list[dict] = []
        self.bound_tools: list[list] = []

    def invoke(self, messages, config=None):
        self.seen.append(list(messages))
        self.configs.append(dict(config or {}))
        return super().invoke(messages, config)

    def bind_tools(self, tools):
        self.bound_tools.append(list(tools))
        return self


def _chain_gateway(openrouter, cloudflare, anthropic):
    """A gateway shaped exactly like the default chain:
    openrouter (primary) -> cloudflare -> anthropic."""
    return ModelGateway(
        primary=None, fallback=None, primary_name="", fallback_name="",
        chain=[
            ProviderEntry("primary", PROVIDER_OPENROUTER, NEMOTRON,
                          openrouter, "prompt"),
            ProviderEntry("fallback_1", PROVIDER_CLOUDFLARE, GLM,
                          cloudflare, "prompt"),
            ProviderEntry("fallback_2", PROVIDER_ANTHROPIC, HAIKU,
                          anthropic, "native"),
        ],
    )


# ---------------------------------------------------------------------------
# 1. OpenRouter success - no fallback
# ---------------------------------------------------------------------------
def test_openrouter_success_uses_primary_only():
    import openai

    primary = _RecordingModel("primary answer")
    cloudflare = _RecordingModel("cloudflare answer")
    anthropic = _RecordingModel("anthropic answer")
    gateway = _chain_gateway(primary, cloudflare, anthropic)

    result = gateway.chat(MESSAGES)

    assert result.content == "primary answer"
    assert primary.calls == 1
    assert cloudflare.calls == 0
    assert anthropic.calls == 0
    assert gateway.last_events == [
        {
            "attempt": "primary",
            "model": NEMOTRON,
            "provider": PROVIDER_OPENROUTER,
            "outcome": "success",
            "duration_ms": primary.__dict__.get("_", 0)
            if False
            else gateway.last_events[0]["duration_ms"],
        }
    ]
    assert gateway.last_events[0]["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# 2.-4. Fallback-eligible primary failures -> Cloudflare
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "primary_exception",
    [
        # HTTP 429 rate limit
        make_openai_error(__import__("openai").RateLimitError, 429),
        # Timeout
        __import__("openai").APITimeoutError(
            request=__import__("httpx2").Request(
                "POST", "https://openrouter.ai/api/v1"
            )
        ),
        # Retryable server error
        make_openai_error(__import__("openai").InternalServerError, 500),
        # Connection failure
        __import__("openai").APIConnectionError(
            request=__import__("httpx2").Request(
                "POST", "https://openrouter.ai/api/v1"
            )
        ),
    ],
    ids=["http-429", "timeout", "server-error-5xx", "connection-failure"],
)
def test_eligible_primary_failure_falls_back_to_cloudflare(primary_exception):
    primary = _RecordingModel(primary_exception)
    cloudflare = _RecordingModel("cloudflare answer")
    anthropic = _RecordingModel("anthropic answer")
    gateway = _chain_gateway(primary, cloudflare, anthropic)

    result = gateway.chat(MESSAGES)

    assert result.content == "cloudflare answer"
    assert primary.calls == 1 and cloudflare.calls == 1
    assert anthropic.calls == 0  # cloudflare succeeded, chain stops there
    # The success event records WHY the fallback happened.
    success = gateway.last_events[-1]
    assert success["provider"] == PROVIDER_CLOUDFLARE
    assert success["fallback_reason"] == (
        f"{PROVIDER_OPENROUTER}_{ErrorCategory.RATE_LIMIT.value}"
    ) or success["fallback_reason"].startswith(PROVIDER_OPENROUTER)


def test_quota_exhaustion_429_falls_back_with_reason():
    import openai

    primary = _RecordingModel(make_openai_error(openai.RateLimitError, 429))
    cloudflare = _RecordingModel("cloudflare answer")
    gateway = _chain_gateway(
        primary, cloudflare, _RecordingModel("anthropic answer")
    )

    result = gateway.chat(MESSAGES)

    assert result.content == "cloudflare answer"
    assert (
        gateway.last_events[-1]["fallback_reason"]
        == "openrouter_rate_limit"
    )
    assert gateway.last_events[0]["category"] == "rate_limit"


# ---------------------------------------------------------------------------
# 5. Cloudflare failure -> legacy Anthropic fallback
# ---------------------------------------------------------------------------
def test_cloudflare_failure_continues_to_legacy_anthropic():
    import openai

    primary = _RecordingModel(make_openai_error(openai.RateLimitError, 429))
    cloudflare = _RecordingModel(
        make_openai_error(openai.InternalServerError, 500)
    )
    anthropic = _RecordingModel("anthropic answer")
    gateway = _chain_gateway(primary, cloudflare, anthropic)

    result = gateway.chat(MESSAGES)

    assert result.content == "anthropic answer"
    assert [e["provider"] for e in gateway.last_events if e["outcome"] == "failed"] == [
        PROVIDER_OPENROUTER, PROVIDER_CLOUDFLARE,
    ]
    assert gateway.last_events[-1]["provider"] == PROVIDER_ANTHROPIC
    assert gateway.last_events[-1]["fallback_reason"] == (
        "cloudflare_transient"
    )


def test_cloudflare_unavailable_is_skipped_not_fatal():
    """A Cloudflare slot without credentials is skipped (event recorded)
    and the chain continues to the legacy fallback."""
    import openai

    primary = _RecordingModel(make_openai_error(openai.RateLimitError, 429))
    anthropic = _RecordingModel("anthropic answer")
    gateway = ModelGateway(
        primary=None, fallback=None, primary_name="", fallback_name="",
        chain=[
            ProviderEntry("primary", PROVIDER_OPENROUTER, NEMOTRON,
                          primary, "prompt"),
            ProviderEntry("fallback_1", PROVIDER_CLOUDFLARE, GLM,
                          None, "prompt"),  # not configured
            ProviderEntry("fallback_2", PROVIDER_ANTHROPIC, HAIKU,
                          anthropic, "native"),
        ],
    )
    result = gateway.chat(MESSAGES)
    assert result.content == "anthropic answer"
    assert any(
        e["outcome"] == "skipped" and e["provider"] == PROVIDER_CLOUDFLARE
        for e in gateway.last_events
    )


# ---------------------------------------------------------------------------
# 6. All providers fail -> descriptive error
# ---------------------------------------------------------------------------
def test_all_providers_fail_raises_descriptive_error():
    primary = _RecordingModel(make_openai_error(openai.RateLimitError, 429))
    cloudflare = _RecordingModel(
        make_openai_error(openai.InternalServerError, 500)
    )
    legacy = _RecordingModel(
        make_anthropic_error(anthropic.InternalServerError, 503)
    )
    gateway = _chain_gateway(primary, cloudflare, legacy)

    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)

    assert len(excinfo.value.attempts) == 3
    message = str(excinfo.value)
    assert PROVIDER_OPENROUTER in message
    assert PROVIDER_CLOUDFLARE in message or "cloudflare" in message
    assert "All configured models failed" in message


def test_all_providers_unconfigured_raises_configuration_error():
    gateway = ModelGateway(
        primary=None, fallback=None, primary_name="", fallback_name="",
        chain=[ProviderEntry("primary", PROVIDER_OPENROUTER, NEMOTRON,
                             None, "prompt")],
    )
    with pytest.raises(ModelGatewayError):
        gateway.chat(MESSAGES)


# ---------------------------------------------------------------------------
# 7. Authentication error on the primary is fallback-eligible
# ---------------------------------------------------------------------------
def test_primary_auth_error_falls_back():
    import openai

    primary = _RecordingModel(make_openai_error(openai.AuthenticationError, 401))
    cloudflare = _RecordingModel("cloudflare answer")
    gateway = _chain_gateway(
        primary, cloudflare, _RecordingModel("anthropic answer")
    )
    result = gateway.chat(MESSAGES)
    assert result.content == "cloudflare answer"


# ---------------------------------------------------------------------------
# 8. Invalid requests: no uncontrolled retries
# ---------------------------------------------------------------------------
def test_invalid_request_on_primary_raises_immediately():
    import openai

    primary = _RecordingModel(make_openai_error(openai.BadRequestError, 400))
    cloudflare = _RecordingModel("cloudflare answer")
    anthropic = _RecordingModel("anthropic answer")
    gateway = _chain_gateway(primary, cloudflare, anthropic)

    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)

    assert excinfo.value.category is ErrorCategory.INVALID_REQUEST
    # No fallback attempts were made.
    assert cloudflare.calls == 0 and anthropic.calls == 0
    assert len(excinfo.value.attempts) == 1


def test_unknown_model_identifier_is_reported_not_substituted():
    import openai

    primary = _RecordingModel(make_openai_error(openai.NotFoundError, 404))
    gateway = _chain_gateway(
        primary, _RecordingModel("cf"), _RecordingModel("ant")
    )
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)
    assert excinfo.value.category is ErrorCategory.INVALID_REQUEST


# ---------------------------------------------------------------------------
# Execution consistency: context and tools survive provider switches
# ---------------------------------------------------------------------------
def test_rag_context_and_tools_reach_the_fallback_provider_identically():
    """The exact message list (with retrieved passages) and the exact tool
    definitions are handed to the fallback attempt (follow-up task 6.5)."""
    primary = _RecordingModel(make_openai_error(openai.RateLimitError, 429))
    cloudflare = _RecordingModel("cloudflare grounded answer")
    gateway = _chain_gateway(
        primary, cloudflare, _RecordingModel("anthropic answer")
    )

    class _Tool:
        name = "search_documents"

    tool = _Tool()
    result = gateway.chat(RAG_MESSAGES, tools=[tool])

    assert result.content == "cloudflare grounded answer"
    # Both attempts received the IDENTICAL message list, passages included.
    assert primary.seen[0] == cloudflare.seen[0] == RAG_MESSAGES
    assert any(
        RAG_PASSAGE in str(m.content) for m in cloudflare.seen[0]
    )
    # The same tools were bound for both attempts.
    assert primary.bound_tools[0] == [tool]
    assert cloudflare.bound_tools[0] == [tool]
    # LangSmith metadata on the fallback attempt records provider, model,
    # and fallback reason (no credentials anywhere).
    cf_config = cloudflare.configs[0]
    assert cf_config["metadata"]["provider"] == PROVIDER_CLOUDFLARE
    assert cf_config["metadata"]["model"] == GLM
    assert cf_config["metadata"]["fallback_reason"] == "openrouter_rate_limit"
    serialized = str(cf_config)
    assert "sk-or" not in serialized and "Bearer" not in serialized


def test_structured_output_falls_back_preserving_messages():
    """Structured-output failures also fail over with the same messages."""
    import json

    from langchain_core.messages import AIMessage

    from core.models import IntentResult

    primary = _RecordingModel(None, structured_result=None)

    def _primary_invoke(messages, config=None):
        import openai

        raise make_openai_error(openai.RateLimitError, 429)

    primary.invoke = _primary_invoke

    class _JsonCloudflare(ScriptedModel):
        def invoke(self, messages, config=None):
            return AIMessage(
                content=json.dumps(
                    {
                        "intent": "general_conversation",
                        "confidence": 0.8,
                    }
                )
            )

    cloudflare = _JsonCloudflare(None)
    gateway = ModelGateway(
        primary=None, fallback=None, primary_name="", fallback_name="",
        chain=[
            ProviderEntry("primary", PROVIDER_OPENROUTER, NEMOTRON,
                          primary, "prompt"),
            ProviderEntry("fallback_1", PROVIDER_CLOUDFLARE, GLM,
                          cloudflare, "prompt"),
        ],
    )
    result = gateway.structured(IntentResult, RAG_MESSAGES)
    assert isinstance(result, IntentResult)
    assert result.intent.value == "general_conversation"


# ---------------------------------------------------------------------------
# Bounded retries
# ---------------------------------------------------------------------------
def test_each_provider_attempted_exactly_once():
    """The chain is the retry bound: no provider is retried, no loops."""
    primary = ScriptedModel(make_openai_error(openai.RateLimitError, 429))
    cloudflare = ScriptedModel(make_openai_error(openai.RateLimitError, 429))
    anthropic = ScriptedModel(make_openai_error(openai.RateLimitError, 429))
    gateway = _chain_gateway(primary, cloudflare, anthropic)

    with pytest.raises(ModelGatewayError):
        gateway.chat(MESSAGES)

    # One attempt per provider, then stop - bounded by the chain length.
    assert (primary.calls, cloudflare.calls, anthropic.calls) == (1, 1, 1)


# ---------------------------------------------------------------------------
# End-to-end: the RAG workflow answers through the Cloudflare fallback slot
# ---------------------------------------------------------------------------
def test_rag_pipeline_answers_through_cloudflare_fallback_position(
    services, settings, ingestion, doc_repo
):
    """The complete document-question workflow survives a provider switch:
    OpenRouter 429s, Cloudflare (prompt-mode structured output) produces
    the grounded answer, and the retrieved passages were in ITS request."""
    import json

    import openai
    from langchain_core.messages import AIMessage

    from core.models import Citation, GroundedAnswer, Intent, IntentResult

    result = ingestion.ingest_file(
        "fallback-notes.txt",
        b"Project Notes\n\nThe vector database is Qdrant.\n\n"
        b"The workflow engine is LangGraph.\n",
    )
    assert result.status == "indexed"

    # Primary: fails the intent classification with a rate limit.
    primary = _RecordingModel(make_openai_error(openai.RateLimitError, 429))

    # Cloudflare slot: prompt-mode structured output over the SAME prompts.
    class _CloudflareScripted(ScriptedModel):
        def __init__(self):
            super().__init__(None)
            self.seen: list[list] = []
            self.calls = 0

        def invoke(self, messages, config=None):
            self.calls += 1
            self.seen.append(list(messages))
            # The prompt-mode gateway appends a SystemMessage with the JSON
            # contract; its schema title identifies which call this is.
            contract = "".join(
                str(getattr(m, "content", "")) for m in messages
            )
            if "GroundedAnswer" in contract:
                # Structured grounded-answer call: schema-valid JSON citing
                # the first source by ordinal.
                reply = {
                    "answer": "The vector database is Qdrant.",
                    "citations": [
                        {
                            "document_id": "irrelevant",
                            "filename": "fallback-notes.txt",
                            "chunk_id": "S1",
                            "supporting_passage": "The vector database is Qdrant.",
                        }
                    ],
                    "confidence": 0.95,
                }
            else:
                # Intent classification call.
                reply = {
                    "intent": "document_question",
                    "confidence": 0.93,
                    "requires_retrieval": True,
                }
            return AIMessage(content=json.dumps(reply))

    cloudflare = _CloudflareScripted()
    services.gateway = ModelGateway(
        primary=None, fallback=None, primary_name="", fallback_name="",
        chain=[
            ProviderEntry("primary", PROVIDER_OPENROUTER, NEMOTRON,
                          primary, "prompt"),
            ProviderEntry("fallback_1", PROVIDER_CLOUDFLARE, GLM,
                          cloudflare, "prompt"),
        ],
    )

    from graph.persistence import create_checkpointer
    from graph.workflow import build_workflow

    graph = build_workflow(
        services, create_checkpointer(settings.checkpoint_db_path)
    )
    thread_id = "rag-through-cloudflare"
    question = "Which vector database does the project use?"
    graph.invoke(
        {
            "user_request": question,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=question)],
        },
        {"configurable": {"thread_id": thread_id}},
    )

    # The workflow completed with a grounded answer produced by the
    # Cloudflare-position model.
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    final = state.values["final_response"]
    assert final["kind"] == "answer"
    assert "Qdrant" in final["text"]
    assert final["citations"], "citations must survive the provider switch"
    # The retrieved passages reached the FALLBACK model's request.
    assert cloudflare.calls >= 2  # intent + grounded answer (at least)
    grounded_calls = [
        m for m in cloudflare.seen
        if "Qdrant" in "".join(str(x.content) for x in m if hasattr(x, "content"))
    ]
    assert grounded_calls, (
        "the retrieved document passages must be in the fallback model's "
        "actual request"
    )
