"""OpenRouter provider integration: provider selection, prompt-based
structured output, tool calling, and fallback for the temporary NVIDIA
Nemotron 3 Ultra primary model.

All tests run offline: real provider clients are only *constructed* (no
network calls), and model behavior is simulated with the scripted fakes.
OpenRouter transport failures surface through the openai SDK (the LangChain
OpenAI integration talks to the OpenRouter endpoint with the same client),
so provider failures are simulated with real openai SDK error objects.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import pytest

from config.settings import Settings


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Defend settings-default tests against a developer's local .env.

    ``app.py`` runs ``load_dotenv()`` at import time; when a UI test
    imports the app first, the local .env's provider variables would leak
    into these default-configuration assertions via the process
    environment.  The shipped project has no .env (git-ignored), but the
    suite must stay green in a developer checkout that has one.
    """
    for var in (
        "LLM_PRIMARY_PROVIDER", "LLM_FALLBACK_PROVIDERS",
        "CLOUDFLARE_ENABLED", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_MODEL", "CLOUDFLARE_STRUCTURED_MODE",
        "OPENROUTER_ENABLED", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "WEB_SEARCH_ENABLED", "WEB_SEARCH_PROVIDER", "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
from core.llm_gateway import (
    ErrorCategory,
    ModelGateway,
    ModelGatewayError,
    ModelOutputError,
    PromptStructuredRunnable,
    parse_model_json,
)
from core.models import (
    Citation,
    GroundedAnswer,
    Intent,
    IntentResult,
    TaskProposal,
)
from graph.nodes import GraphServices
from graph.persistence import create_checkpointer
from graph.workflow import build_workflow
from tests.fakes import (
    ScriptedModel,
    SequentialTextModel,
    make_openai_error,
    tool_call_message,
)

import openai

MESSAGES = [HumanMessage(content="hello")]

NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Offline test credential used ONLY to construct clients; never sent anywhere.
OPENROUTER_TEST_KEY = "sk-or-test-construction-only"

INTENT_JSON = json.dumps(
    {"intent": "general_conversation", "confidence": 0.9}
)
PROPOSAL_JSON = json.dumps(
    {
        "title": "Review lecture notes",
        "description": "Go over the ML lecture notes before Friday.",
        "due_date": None,
        "priority": "medium",
        "confirmation_message": (
            "I will create the task 'Review lecture notes' (medium priority)."
        ),
    }
)


def _openrouter_gateway(primary, fallback=None, **overrides) -> ModelGateway:
    """A gateway shaped exactly like LLM_PRIMARY_PROVIDER=openrouter builds it."""
    kwargs = dict(
        primary=primary,
        fallback=fallback,
        primary_name=NEMOTRON,
        fallback_name="claude-haiku-5-5",
        primary_provider="openrouter",
        fallback_provider="anthropic",
        primary_structured_mode="prompt",
        fallback_structured_mode="native",
    )
    kwargs.update(overrides)
    return ModelGateway(**kwargs)


# ---------------------------------------------------------------------------
# A. Provider initialization (construction only - no network calls)
# ---------------------------------------------------------------------------
class _OpenRouterSettings:
    """Duck-typed settings matching LLM_PRIMARY_PROVIDER=openrouter."""

    llm_primary_provider = "openrouter"
    openrouter_api_key = OPENROUTER_TEST_KEY
    openrouter_base_url = OPENROUTER_URL
    openrouter_model = NEMOTRON
    openai_api_key = "sk-test-original"
    anthropic_api_key = "sk-ant-test"
    primary_model = "gpt-6-luna"
    fallback_model = "claude-haiku-5-5"
    max_output_tokens = 1500
    model_timeout_seconds = 30.0


def test_from_settings_openrouter_builds_primary_with_nemotron():
    gateway = ModelGateway.from_settings(_OpenRouterSettings())
    info = gateway.describe()
    assert info["primary_available"] is True
    assert info["primary_model"] == NEMOTRON
    assert info["primary_provider"] == "openrouter"
    assert info["primary_structured_output"] == "prompt"
    # The Anthropic fallback is preserved unchanged.
    assert info["fallback_available"] is True
    assert info["fallback_model"] == "claude-haiku-5-5"
    assert info["fallback_provider"] == "anthropic"
    assert info["fallback_structured_output"] == "native"


def test_openrouter_primary_targets_chat_completions_endpoint():
    from langchain_openai import ChatOpenAI

    gateway = ModelGateway.from_settings(_OpenRouterSettings())
    primary = gateway._primary
    assert isinstance(primary, ChatOpenAI)
    assert primary.model_name == NEMOTRON
    assert str(primary.openai_api_base).rstrip("/") == OPENROUTER_URL
    # The chat-completions API must be used, not the Responses API.
    assert primary.use_responses_api is False
    assert type(primary.root_client).__name__ == "OpenAI"
    assert str(primary.root_client.base_url).rstrip("/") == OPENROUTER_URL
    # Attribution header is set; no credentials leak into headers.
    assert primary.default_headers.get("X-Title") == (
        "personal-knowledge-task-assistant"
    )


def test_openrouter_missing_key_disables_primary_but_preserves_fallback():
    class _NoKey(_OpenRouterSettings):
        openrouter_api_key = None

    gateway = ModelGateway.from_settings(_NoKey())
    info = gateway.describe()
    assert info["primary_available"] is False
    assert info["primary_model"] == NEMOTRON  # still reported for the UI
    assert info["primary_provider"] == "openrouter"
    # The Anthropic fallback remains fully configured.
    assert info["fallback_available"] is True


def test_openrouter_missing_key_skips_to_fallback_with_event():
    fallback = ScriptedModel("fallback answer")
    gateway = _openrouter_gateway(None, fallback)
    result = gateway.chat(MESSAGES)
    assert result.content == "fallback answer"
    assert any(
        event["outcome"] == "skipped" and event["provider"] == "openrouter"
        for event in gateway.last_events
    )


def test_original_openai_provider_restored_by_configuration_only():
    """Switching back is a configuration change, not a code change."""

    class _OpenAiSettings(_OpenRouterSettings):
        llm_primary_provider = "openai"
        openrouter_api_key = OPENROUTER_TEST_KEY

    gateway = ModelGateway.from_settings(_OpenAiSettings())
    info = gateway.describe()
    assert info["primary_provider"] == "openai"
    assert info["primary_model"] == "gpt-6-luna"
    assert info["primary_structured_output"] == "native"
    from langchain_openai import ChatOpenAI

    assert isinstance(gateway._primary, ChatOpenAI)
    assert gateway._primary.use_responses_api is True


def test_unknown_provider_is_reported_never_substituted():
    class _BadProvider(_OpenRouterSettings):
        llm_primary_provider = "martian"

    gateway = ModelGateway.from_settings(_BadProvider())
    info = gateway.describe()
    assert info["primary_available"] is False
    assert info["primary_provider"] == "martian"


# ---------------------------------------------------------------------------
# Settings: provider selection, validation, and env wiring
# ---------------------------------------------------------------------------
def test_settings_default_provider_is_cloudflare_with_openrouter_fallback():
    """Follow-up task 4, section 5: Cloudflare GLM-4.7-Flash is the default
    primary; OpenRouter Nemotron 3 Ultra is the default first fallback."""
    settings = Settings(_env_file=None)
    assert settings.llm_primary_provider == "cloudflare"
    assert settings.llm_fallback_providers == "openrouter"
    assert settings.openrouter_model == NEMOTRON
    assert settings.openrouter_base_url == OPENROUTER_URL
    assert settings.active_primary_model == "@cf/zai-org/glm-4.7-flash"
    assert settings.openrouter_enabled is True


def test_settings_env_variables_select_openrouter(monkeypatch):
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_TEST_KEY)
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/some-other-model:free")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/")
    settings = Settings(_env_file=None)
    assert settings.llm_primary_provider == "openrouter"
    assert settings.openrouter_model == "nvidia/some-other-model:free"
    assert settings.active_primary_model == "nvidia/some-other-model:free"
    assert settings.active_primary_available is True


def test_settings_validate_openrouter_without_key_is_an_error():
    settings = Settings(_env_file=None, llm_primary_provider="openrouter")
    errors, _warnings = settings.validate()
    assert any("OPENROUTER_API_KEY" in error for error in errors)


def test_settings_validate_openrouter_with_key_warns_about_fallback_only():
    settings = Settings(
        _env_file=None,
        llm_primary_provider="openrouter",
        openrouter_api_key=OPENROUTER_TEST_KEY,
        openai_api_key="sk-orig",
    )
    errors, warnings = settings.validate()
    assert not any("OPENROUTER_API_KEY" in error for error in errors)
    assert any("ANTHROPIC_API_KEY" in warning for warning in warnings)
    assert any("not active" in warning for warning in warnings)


def test_settings_validate_unknown_provider_is_an_error():
    settings = Settings(_env_file=None, llm_primary_provider="martian")
    errors, _warnings = settings.validate()
    assert any("LLM_PRIMARY_PROVIDER" in error for error in errors)


def test_settings_validate_openai_path_unchanged():
    """The original openai-primary path still validates cleanly (selected
    explicitly; the default primary is Cloudflare since follow-up task 4).
    The OpenRouter first fallback is configured so the chain is
    warning-free."""
    settings = Settings(
        _env_file=None,
        llm_primary_provider="openai",
        openai_api_key="sk-test",
        openrouter_api_key="or-test",
        anthropic_api_key="sk-ant-test",
        # Web search stays out of this test's scope: without a Tavily key
        # it would (correctly) warn about the optional tool being
        # unavailable (follow-up task 4, section 9).
        web_search_enabled=False,
    )
    errors, warnings = settings.validate()
    assert errors == [] and warnings == []
    summary = settings.public_summary()
    assert summary["primary_model"] == "gpt-6-luna"
    assert summary["llm_primary_provider"] == "openai"
    assert summary["primary_available"] is True


def test_settings_default_primary_requires_cloudflare_credentials():
    """Follow-up task 4, section 5: with the Cloudflare primary selected by
    default, missing credentials are a clear startup ERROR (never a silent
    skip) naming the missing variables."""
    settings = Settings(_env_file=None)
    errors, warnings = settings.validate()
    assert any(
        "LLM_PRIMARY_PROVIDER is 'cloudflare'" in e
        and "CLOUDFLARE_ACCOUNT_ID" in e
        and "CLOUDFLARE_API_TOKEN" in e
        for e in errors
    )


def test_settings_warns_when_chain_lacks_cloudflare_credentials():
    """Cloudflare in the FALLBACK chain but unconfigured produces a clear
    warning, never a silent skip (unchanged behaviour; the default chain
    since follow-up task 4 is cloudflare-primary -> openrouter -> ...)."""
    settings = Settings(
        _env_file=None,
        llm_primary_provider="openai",
        llm_fallback_providers="cloudflare",
        openai_api_key="sk-test",
        anthropic_api_key="sk-ant-test",
    )
    errors, warnings = settings.validate()
    assert errors == []
    assert any(
        "CLOUDFLARE_ACCOUNT_ID" in w and "CLOUDFLARE_API_TOKEN" in w
        for w in warnings
    )


def test_public_summary_reports_active_openrouter_configuration():
    settings = Settings(
        _env_file=None,
        llm_primary_provider="openrouter",
        openrouter_api_key=OPENROUTER_TEST_KEY,
    )
    summary = settings.public_summary()
    assert summary["primary_model"] == NEMOTRON
    assert summary["openai_primary_model"] == "gpt-6-luna"
    assert summary["llm_primary_provider"] == "openrouter"
    assert summary["primary_available"] is True


# ---------------------------------------------------------------------------
# Robust JSON parsing (Markdown fences, prose, malformed replies)
# ---------------------------------------------------------------------------
def test_parse_model_json_plain():
    assert parse_model_json('{"a": 1}') == {"a": 1}


def test_parse_model_json_fenced_with_language_tag():
    text = '```json\n{"a": 1}\n```'
    assert parse_model_json(text) == {"a": 1}


def test_parse_model_json_fenced_without_language_tag():
    text = '```\n{"a": {"b": [1, 2]}}\n```'
    assert parse_model_json(text) == {"a": {"b": [1, 2]}}


def test_parse_model_json_embedded_in_prose():
    text = (
        "Sure! Here is the JSON you asked for:\n"
        '{"intent": "general_conversation", "confidence": 0.9}\n'
        "Hope that helps."
    )
    parsed = parse_model_json(text)
    assert parsed["intent"] == "general_conversation"


def test_parse_model_json_handles_braces_inside_strings():
    text = '{"answer": "use {curly} braces freely", "confidence": 0.5}'
    assert parse_model_json(text)["answer"] == "use {curly} braces freely"


def test_parse_model_json_empty_or_garbage_raises():
    with pytest.raises(ValueError):
        parse_model_json("")
    with pytest.raises(ValueError):
        parse_model_json("   ")
    with pytest.raises(ValueError):
        parse_model_json("I cannot answer that in JSON, sorry.")


# ---------------------------------------------------------------------------
# Prompt-based structured output (OpenRouter / Nemotron compatibility)
# ---------------------------------------------------------------------------
def test_prompt_structured_plain_json_success():
    primary = SequentialTextModel([INTENT_JSON])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert result.intent is Intent.GENERAL_CONVERSATION
    assert primary.calls == 1  # no correction needed


def test_prompt_structured_sends_json_contract_and_schema():
    primary = SequentialTextModel([INTENT_JSON])
    gateway = _openrouter_gateway(primary)
    gateway.structured(IntentResult, MESSAGES, run_name="classify_intent")
    payload = primary.received[0]
    contract = next(
        m for m in payload if isinstance(m, SystemMessage)
    )
    assert "OUTPUT FORMAT REQUIREMENTS" in contract.content
    assert '"general_conversation"' in contract.content  # enum documented
    assert "IntentResult" in str(payload) or "intent" in contract.content


def test_prompt_structured_accepts_fenced_json():
    primary = SequentialTextModel([f"```json\n{INTENT_JSON}\n```"])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)


def test_prompt_structured_accepts_json_with_surrounding_prose():
    reply = f"Here is my answer:\n{INTENT_JSON}\nLet me know if unclear."
    primary = SequentialTextModel([reply])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)


def test_prompt_structured_correction_recovers_from_invalid_enum():
    bad = json.dumps({"intent": "chitchat", "confidence": 0.9})
    primary = SequentialTextModel([bad, INTENT_JSON])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert primary.calls == 2  # one bounded correction attempt
    # The correction turn shows the model its invalid reply and the problem.
    correction_payload = primary.received[1]
    feedback = [str(m.content) for m in correction_payload]
    assert any("not valid" in text for text in feedback)
    assert any("chitchat" in text or "intent" in text for text in feedback)


def test_prompt_structured_extra_fields_rejected_then_corrected():
    extra = json.dumps(
        {
            "intent": "general_conversation",
            "confidence": 0.9,
            "reasoning": "because I say so",
        }
    )
    primary = SequentialTextModel([extra, INTENT_JSON])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert primary.calls == 2


def test_prompt_structured_persistent_malformed_json_falls_back():
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    primary = SequentialTextModel(
        ["no json here at all", "still no json"]
    )
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert primary.calls == 2  # initial + one bounded correction
    assert fallback.structured_calls == 1
    assert gateway.last_events[0]["outcome"] == "failed"
    assert gateway.last_events[0]["category"] == "model_output"
    assert gateway.last_events[0]["provider"] == "openrouter"


def test_prompt_structured_persistent_schema_violations_fall_back():
    primary = SequentialTextModel(
        [
            '{"intent": "bogus", "confidence": 0.9}',
            '{"intent": "still_bogus", "confidence": 0.9}',
        ]
    )
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert fallback.structured_calls == 1


def test_prompt_structured_never_accepts_invalid_output_without_fallback():
    primary = SequentialTextModel(["garbage", "garbage again"])
    gateway = _openrouter_gateway(primary, fallback=None)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.structured(IntentResult, MESSAGES)
    # No fallback is configured, so the aggregate error is raised; the
    # per-attempt bookkeeping shows the failure was a model-output problem
    # (invalid structured output is never accepted as valid).
    assert "not produce valid IntentResult JSON" in str(excinfo.value)
    assert excinfo.value.attempts[0]["category"] == "model_output"
    assert excinfo.value.attempts[0]["provider"] == "openrouter"


def test_prompt_structured_empty_reply_falls_back():
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    primary = SequentialTextModel(["", ""])
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert fallback.structured_calls == 1


def test_prompt_structured_tool_call_response_is_rejected():
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    primary = SequentialTextModel(
        [tool_call_message("search_documents", {"query": "hi"})]
    )
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert fallback.structured_calls == 1


def test_prompt_structured_task_proposal_schema():
    primary = SequentialTextModel([f"```json\n{PROPOSAL_JSON}\n```"])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(TaskProposal, MESSAGES)
    assert isinstance(result, TaskProposal)
    assert result.title == "Review lecture notes"
    assert result.priority.value == "medium"


def test_prompt_structured_task_proposal_rejects_past_due_date():
    invalid = json.dumps(
        {
            "title": "Past due task",
            "confirmation_message": "ok",
            "due_date": "2000-01-01",
        }
    )
    good = TaskProposal(title="Valid task", confirmation_message="ok")
    primary = SequentialTextModel([invalid, invalid])
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.structured(TaskProposal, MESSAGES)
    # The prompt-mode model never produced a valid proposal: the fallback
    # did. Invalid structured output is never accepted as valid.
    assert isinstance(result, TaskProposal)
    assert result.title == "Valid task"
    assert fallback.structured_calls == 1


def test_prompt_structured_grounded_answer_schema():
    payload = json.dumps(
        {
            "answer": "According to your notes, Qdrant is used.",
            "citations": [
                {
                    "document_id": "doc-1",
                    "filename": "notes.txt",
                    "page_number": None,
                    "section": None,
                    "chunk_id": "doc-1:0",
                    "supporting_passage": "Qdrant is used.",
                    "score": 0.9,
                }
            ],
            "confidence": 0.85,
            "insufficient_evidence": False,
        }
    )
    primary = SequentialTextModel([payload])
    gateway = _openrouter_gateway(primary)
    result = gateway.structured(GroundedAnswer, MESSAGES)
    assert isinstance(result, GroundedAnswer)
    assert result.citations[0].chunk_id == "doc-1:0"


def test_prompt_structured_runnable_labels_correction_in_config():
    seen_configs: list[dict | None] = []

    class _Recording(SequentialTextModel):
        def invoke(self, messages, config=None):
            seen_configs.append(config)
            return super().invoke(messages, config)

    primary = _Recording(["not json", INTENT_JSON])
    runnable = PromptStructuredRunnable(primary, IntentResult)
    result = runnable.invoke(MESSAGES, config={"run_name": "classify_intent"})
    assert isinstance(result, IntentResult)
    assert seen_configs[0] is None or seen_configs[0].get("run_name") in (
        None,
        "classify_intent",
    )
    assert seen_configs[1]["run_name"] == "classify_intent_correction"
    assert seen_configs[1]["metadata"]["correction_attempt"] == 1


# ---------------------------------------------------------------------------
# Tool calling through the OpenRouter slot
# ---------------------------------------------------------------------------
def test_chat_binds_tools_through_openrouter_slot():
    primary = SequentialTextModel(
        [tool_call_message("search_documents", {"query": "qdrant"})]
    )
    gateway = _openrouter_gateway(primary)
    response = gateway.chat(MESSAGES, tools=["search_documents"])
    assert response.tool_calls[0]["name"] == "search_documents"
    assert gateway.last_events[-1]["provider"] == "openrouter"


def test_chat_distinguishes_plain_answer_from_tool_call():
    primary = SequentialTextModel(["Plain answer, no tools."])
    gateway = _openrouter_gateway(primary)
    response = gateway.chat(MESSAGES, tools=["search_documents"])
    assert not response.tool_calls
    assert response.content == "Plain answer, no tools."


def test_chat_malformed_tool_arguments_surface_as_errors_to_the_loop():
    # A tool call with broken arguments is returned as-is; the graph node
    # invokes the tool with them, catches the failure, and feeds it back.
    primary = SequentialTextModel(
        [tool_call_message("search_documents", {"bogus": "value"})]
    )
    gateway = _openrouter_gateway(primary)
    response = gateway.chat(MESSAGES, tools=["search_documents"])
    assert response.tool_calls[0]["args"] == {"bogus": "value"}


# ---------------------------------------------------------------------------
# Fallback behavior for OpenRouter transport failures
# (OpenRouter errors surface through the openai SDK client)
# ---------------------------------------------------------------------------
def _openrouter_error(error_class, status):
    return make_openai_error(
        error_class, status
    )  # endpoint host differs but the SDK class is identical


@pytest.mark.parametrize(
    "primary_error",
    [
        make_openai_error(openai.RateLimitError, 429),
        make_openai_error(openai.AuthenticationError, 401),
        make_openai_error(openai.InternalServerError, 500),
    ],
)
def test_openrouter_eligible_failure_falls_back(primary_error):
    primary = SequentialTextModel([primary_error])
    fallback = ScriptedModel("fallback answer")
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.chat(MESSAGES)
    assert result.content == "fallback answer"
    assert gateway.last_events[0]["provider"] == "openrouter"
    assert gateway.last_events[0]["outcome"] == "failed"
    assert gateway.last_events[1]["provider"] == "anthropic"
    assert gateway.last_events[1]["outcome"] == "success"


def test_openrouter_connection_timeout_falls_back():
    import httpx2

    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    primary = SequentialTextModel(
        [openai.APITimeoutError(request=request)]
    )
    fallback = ScriptedModel("fallback answer")
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.chat(MESSAGES)
    assert result.content == "fallback answer"


def test_openrouter_unknown_model_never_falls_back():
    error = make_openai_error(openai.NotFoundError, 404)
    primary = SequentialTextModel([error])
    fallback = ScriptedModel("should not be used")
    gateway = _openrouter_gateway(primary, fallback)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)
    assert excinfo.value.category is ErrorCategory.INVALID_REQUEST
    assert fallback.calls == 0


def test_openrouter_structured_failure_falls_back_to_native_fallback():
    error = make_openai_error(openai.RateLimitError, 429)
    primary = SequentialTextModel([error])
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _openrouter_gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert fallback.structured_calls == 1


def test_openrouter_unavailable_and_no_fallback_reports_clearly():
    error = make_openai_error(openai.RateLimitError, 429)
    primary = SequentialTextModel([error])
    gateway = _openrouter_gateway(primary, fallback=None)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)
    assert "nemotron" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Full-workflow verification with an OpenRouter-shaped gateway
# (LangGraph execution, RAG citations, human approval - all offline)
# ---------------------------------------------------------------------------
CHUNK_TEXT = "Qdrant was selected as the vector database for the project."


@pytest.fixture
def openrouter_graph(tmp_path, settings, retriever, task_repo, memory_repo, conv_repo):
    """The real workflow wired to a real ModelGateway whose primary slot is
    configured exactly like the OpenRouter provider (prompt-mode structured
    output), with no fallback so every call must succeed through it."""

    def _build(replies: list):
        primary = SequentialTextModel(replies)
        gateway = _openrouter_gateway(primary, fallback=None)
        services = GraphServices(
            gateway=gateway,
            retriever=retriever,
            task_repo=task_repo,
            memory_repo=memory_repo,
            conversation_repo=conv_repo,
            settings=settings,
        )
        checkpointer = create_checkpointer(settings.checkpoint_db_path)
        return build_workflow(services, checkpointer), primary

    return _build


def _invoke(graph, thread_id, request):
    return graph.invoke(
        {
            "user_request": request,
            "thread_id": thread_id,
            "messages": [HumanMessage(content=request)],
        },
        {"configurable": {"thread_id": thread_id}},
    )


def test_workflow_rag_answer_with_citations_via_openrouter(
    openrouter_graph, ingestion, doc_repo
):
    result = ingestion.ingest_file("plan.txt", CHUNK_TEXT.encode())
    assert result.status == "indexed"
    document = doc_repo.list_documents()[0]
    chunk_id = f"{document.document_id}:0"

    grounded = json.dumps(
        {
            "answer": "Qdrant was selected as the vector database.",
            "citations": [
                {
                    "document_id": document.document_id,
                    "filename": "plan.txt",
                    "chunk_id": chunk_id,
                    "supporting_passage": CHUNK_TEXT,
                }
            ],
            "confidence": 0.9,
            "insufficient_evidence": False,
        }
    )
    replies = [
        f"```json\n{json.dumps({'intent': 'document_question', 'confidence': 0.95})}\n```",
        "DONE",  # retrieval planner: no tool call -> direct retrieval
        f"Here is the grounded answer:\n{grounded}",
    ]
    graph, primary = openrouter_graph(replies)
    outcome = _invoke(graph, "t-or-rag", CHUNK_TEXT)

    payload = outcome["final_response"]
    assert payload["kind"] == "answer"
    assert payload["citations"][0]["chunk_id"] == chunk_id
    assert payload["insufficient_evidence"] is False
    # Every model call went through the OpenRouter-shaped primary slot:
    # intent classification (prompt JSON), retrieval planner (plain chat),
    # and the grounded answer (prompt JSON with fences + prose).
    assert primary.calls == 3


def test_workflow_human_approval_resume_via_openrouter(openrouter_graph, task_repo):
    from langgraph.types import Command

    replies = [
        json.dumps({"intent": "create_task", "confidence": 0.95}),
        f"```json\n{PROPOSAL_JSON}\n```",
    ]
    graph, primary = openrouter_graph(replies)

    outcome = _invoke(graph, "t-or-approval", "Add a task to review lecture notes")
    interrupts = outcome.get("__interrupt__") or ()
    assert interrupts, "workflow must pause at human_approval"
    payload = interrupts[0].value
    assert payload["type"] == "task_approval"
    assert payload["proposal"]["title"] == "Review lecture notes"

    # Reject first: nothing is saved, and the thread stays resumable-free.
    outcome = graph.invoke(
        Command(resume="rejected"), {"configurable": {"thread_id": "t-or-approval"}}
    )
    assert outcome["final_response"]["kind"] == "task_rejected"
    assert task_repo.list_tasks() == []

    # New turn: approve this time; the task is saved exactly once.
    replies2 = [
        json.dumps({"intent": "create_task", "confidence": 0.95}),
        f"```json\n{PROPOSAL_JSON}\n```",
    ]
    primary.replies.extend(replies2)
    outcome = _invoke(graph, "t-or-approval2", "Please add the review task")
    interrupts = outcome.get("__interrupt__") or ()
    assert interrupts
    outcome = graph.invoke(
        Command(resume="approved"), {"configurable": {"thread_id": "t-or-approval2"}}
    )
    assert outcome["final_response"]["kind"] == "task_created"
    tasks = task_repo.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].title == "Review lecture notes"

    # A duplicate resume on the already-resolved thread must not duplicate
    # the task (some LangGraph versions reject such resumes outright).
    try:
        graph.invoke(
            Command(resume="approved"),
            {"configurable": {"thread_id": "t-or-approval2"}},
        )
    except Exception:  # noqa: BLE001 - rejection is acceptable here
        pass
    assert task_repo.count_tasks() == 1
