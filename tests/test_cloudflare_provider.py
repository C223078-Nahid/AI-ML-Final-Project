"""Cloudflare Workers AI provider tests (follow-up task, section 6).

All tests run offline: provider construction is verified through the
constructed ChatOpenAI client's attributes (model id, base URL, API key
presence - never its value), and error classification through the typed
SDK exceptions the OpenAI-compatible endpoint surfaces.

The REAL endpoint compatibility check (an actual GLM-4.7-Flash request)
lives in ``scripts/probe_cloudflare.py`` and runs when credentials are
available - see the README.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from config.settings import Settings
from core.llm_gateway import (
    DEFAULT_CLOUDFLARE_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_CLOUDFLARE,
    PROVIDER_OPENROUTER,
    ErrorCategory,
    ModelGateway,
    ProviderEntry,
    classify_exception,
)
from tests.fakes import ScriptedModel

MESSAGES = [HumanMessage(content="hello")]

CF_ACCOUNT = "deadbeefcafe1234567890abcdef1234"
CF_TOKEN = "cf-api-token-for-tests"
NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b:free"


class _IntentLike(BaseModel):
    answer: str


def _cf_settings(**overrides) -> Settings:
    """OpenRouter primary + configured Cloudflare fallback + Anthropic."""
    defaults = dict(
        _env_file=None,
        llm_primary_provider="openrouter",
        openrouter_api_key="or-key",
        anthropic_api_key="ant-key",
        cloudflare_enabled=True,
        cloudflare_account_id=CF_ACCOUNT,
        cloudflare_api_token=CF_TOKEN,
        cloudflare_model="@cf/zai-org/glm-4.7-flash",
        cloudflare_base_url="https://api.cloudflare.com/client/v4/accounts",
        llm_fallback_providers="cloudflare",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# A. Configuration loading
# ---------------------------------------------------------------------------
def test_settings_load_cloudflare_defaults():
    settings = Settings(_env_file=None)
    assert settings.cloudflare_enabled is True
    assert settings.cloudflare_model == "@cf/zai-org/glm-4.7-flash"
    assert (
        settings.cloudflare_base_url
        == "https://api.cloudflare.com/client/v4/accounts"
    )
    # Follow-up task 4, section 5: Cloudflare is the default PRIMARY and
    # OpenRouter (Nemotron) the default first fallback.
    assert settings.llm_primary_provider == "cloudflare"
    assert settings.llm_fallback_providers == "openrouter"
    assert settings.cloudflare_structured_mode == "prompt"


def test_settings_base_url_construction_from_parts():
    settings = _cf_settings()
    assert settings.cloudflare_chat_base_url == (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CF_ACCOUNT}/ai/v1"
    )


def test_settings_base_url_construction_custom_prefix():
    settings = _cf_settings(
        cloudflare_base_url="https://api.cloudflare.com/client/v4/accounts/"
    )
    assert settings.cloudflare_chat_base_url == (
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/v1"
    )


def test_missing_account_id_disables_cloudflare():
    settings = _cf_settings(cloudflare_account_id=None)
    assert settings.cloudflare_available is False
    errors, warnings = settings.validate()
    assert errors == []
    assert any("CLOUDFLARE_ACCOUNT_ID" in w for w in warnings)


def test_missing_api_token_disables_cloudflare():
    settings = _cf_settings(cloudflare_api_token=None)
    assert settings.cloudflare_available is False
    errors, warnings = settings.validate()
    assert errors == []
    assert any("CLOUDFLARE_API_TOKEN" in w for w in warnings)


def test_cloudflare_disabled_flag_removes_provider():
    settings = _cf_settings(cloudflare_enabled=False)
    assert settings.cloudflare_available is False
    chain = settings.fallback_provider_chain()
    assert PROVIDER_CLOUDFLARE in chain  # stays configured in the list...
    # ...but the gateway marks it unavailable (skipped at request time).
    gateway = ModelGateway.from_settings(settings)
    providers = [e["provider"] for e in gateway.describe()["chain"]]
    assert PROVIDER_CLOUDFLARE in providers
    entry = next(
        e for e in gateway._chain if e.provider == PROVIDER_CLOUDFLARE
    )
    assert entry.available is False


def test_cloudflare_as_primary_requires_credentials():
    settings = _cf_settings(
        llm_primary_provider="cloudflare",
        cloudflare_api_token=None,
    )
    errors, _ = settings.validate()
    assert any("CLOUDFLARE_API_TOKEN" in e for e in errors)


# ---------------------------------------------------------------------------
# B. Gateway construction (no network)
# ---------------------------------------------------------------------------
def test_gateway_builds_cloudflare_client_with_exact_model_and_url():
    from langchain_openai import ChatOpenAI

    gateway = ModelGateway.from_settings(_cf_settings())
    entry = next(
        e for e in gateway._chain if e.provider == PROVIDER_CLOUDFLARE
    )
    model = entry.model
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "@cf/zai-org/glm-4.7-flash"
    assert str(model.openai_api_base).rstrip("/") == (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CF_ACCOUNT}/ai/v1"
    )
    # The token flows into the client; never into logs, metadata, or traces.
    assert model.openai_api_key.get_secret_value() == CF_TOKEN
    # OpenAI-compatible chat completions, not the Responses API.
    assert model.use_responses_api is False
    assert entry.structured_mode == "prompt"


def test_gateway_chain_order_openrouter_cloudflare_anthropic():
    gateway = ModelGateway.from_settings(_cf_settings())
    chain = gateway.describe()["chain"]
    assert [(e["role"], e["provider"]) for e in chain] == [
        ("primary", PROVIDER_OPENROUTER),
        ("fallback_1", PROVIDER_CLOUDFLARE),
        ("fallback_2", PROVIDER_ANTHROPIC),
    ]
    assert all(e["available"] for e in chain)


def test_gateway_chain_respects_configured_order():
    settings = _cf_settings(llm_fallback_providers="anthropic,cloudflare")
    gateway = ModelGateway.from_settings(settings)
    providers = [e["provider"] for e in gateway.describe()["chain"][1:]]
    # Configured order first (anthropic), then dedup keeps cloudflare second;
    # the legacy anthropic is NOT appended twice.
    assert providers == [PROVIDER_ANTHROPIC, PROVIDER_CLOUDFLARE]


def test_gateway_chain_skips_unconfigured_entries_with_event():
    settings = _cf_settings(
        openrouter_api_key=None, cloudflare_api_token=None,
        anthropic_api_key=None,
    )
    gateway = ModelGateway.from_settings(settings)
    with pytest.raises(Exception):
        gateway.chat(MESSAGES)
    skipped = [e for e in gateway.last_events if e["outcome"] == "skipped"]
    assert {e["provider"] for e in skipped} == {
        PROVIDER_OPENROUTER, PROVIDER_CLOUDFLARE, PROVIDER_ANTHROPIC,
    }


def test_unknown_fallback_provider_is_rejected_at_validation():
    settings = _cf_settings(llm_fallback_providers="martian")
    errors, _ = settings.validate()
    assert any("martian" in e and "LLM_FALLBACK_PROVIDERS" in e
               for e in errors)


def test_default_model_identifier_constant():
    assert DEFAULT_CLOUDFLARE_MODEL == "@cf/zai-org/glm-4.7-flash"


# ---------------------------------------------------------------------------
# C. Error classification (OpenAI SDK exceptions wrap Cloudflare responses)
# ---------------------------------------------------------------------------
def test_rate_limit_maps_to_fallback_eligible():
    import openai

    exc = openai.RateLimitError(
        "quota exhausted", response=_fake_response(429), body=None
    )
    assert classify_exception(exc) is ErrorCategory.RATE_LIMIT


def test_timeout_and_connection_map_to_transient():
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.cloudflare.com/x")
    timeout = openai.APITimeoutError(request=request)
    assert classify_exception(timeout) is ErrorCategory.TRANSIENT
    connection = openai.APIConnectionError(
        request=request, message="connection dropped"
    )
    assert classify_exception(connection) is ErrorCategory.TRANSIENT


def test_server_error_maps_to_transient():
    import openai

    exc = openai.InternalServerError(
        "worker crashed", response=_fake_response(500), body=None
    )
    assert classify_exception(exc) is ErrorCategory.TRANSIENT


def test_auth_error_is_eligible_for_fallback():
    import openai

    exc = openai.AuthenticationError(
        "invalid token", response=_fake_response(401), body=None
    )
    assert classify_exception(exc) is ErrorCategory.AUTH


def test_invalid_request_does_not_trigger_fallback():
    import openai

    exc = openai.BadRequestError(
        "unknown model", response=_fake_response(400), body=None
    )
    assert classify_exception(exc) is ErrorCategory.INVALID_REQUEST


def test_not_found_does_not_trigger_fallback():
    import openai

    exc = openai.NotFoundError(
        "no such route", response=_fake_response(404), body=None
    )
    assert classify_exception(exc) is ErrorCategory.INVALID_REQUEST


def _fake_response(status_code: int):
    import httpx

    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.cloudflare.com/x"),
        json={"success": False, "errors": [{"code": status_code}]},
    )


# ---------------------------------------------------------------------------
# D. Error enrichment (clear configuration errors)
# ---------------------------------------------------------------------------
def test_cloudflare_invalid_request_error_carries_configuration_hint():
    """Invalid credentials / invalid model identifiers on Cloudflare are
    reported as a clear configuration error (with a hint), never as an
    uncontrolled retry storm."""
    import openai

    # The primary fails with a rate limit (fallback-eligible) so the chain
    # reaches Cloudflare; Cloudflare then fails with a hard config error.
    rate_limited = openai.RateLimitError(
        "openrouter quota exhausted", response=_fake_response(429), body=None
    )
    primary = ScriptedModel(rate_limited)

    class _RaisingCloudflare(ScriptedModel):
        def invoke(self, *a, **k):
            raise openai.BadRequestError(
                "model not found", response=_fake_response(400), body=None
            )

    gateway = ModelGateway(
        primary=primary,
        fallback=_RaisingCloudflare("x"),
        primary_name=NEMOTRON,
        fallback_name="@cf/zai-org/glm-4.7-flash",
        primary_provider=PROVIDER_OPENROUTER,
        fallback_provider=PROVIDER_CLOUDFLARE,
        primary_structured_mode="prompt",
        fallback_structured_mode="prompt",
    )
    with pytest.raises(Exception) as excinfo:
        gateway.chat(MESSAGES)
    message = str(excinfo.value)
    # A clear configuration error mentioning the Cloudflare settings.
    assert "cloudflare" in message.lower()
    assert "CLOUDFLARE_MODEL" in message or "CLOUDFLARE_ACCOUNT_ID" in message
    # INVALID_REQUEST is not fallback-eligible: exactly two attempts.
    assert len(excinfo.value.attempts) == 2


# ---------------------------------------------------------------------------
# E. Structured output through the Cloudflare slot (prompt mode)
# ---------------------------------------------------------------------------
def test_structured_output_prompt_mode_through_cloudflare_entry():
    """Prompt-mode structured output (the Cloudflare default) parses model
    JSON and validates it against the Pydantic schema."""
    import json

    from langchain_core.messages import AIMessage

    from core.models import IntentResult

    class _JsonModel(ScriptedModel):
        def invoke(self, messages, config=None, **kwargs):
            payload = {
                "intent": "general_conversation",
                "confidence": 0.87,
            }
            return AIMessage(content=json.dumps(payload))

    entry_model = _JsonModel(None)
    gateway = ModelGateway(
        primary=None,
        fallback=None,
        primary_name="",
        fallback_name="",
        chain=[
            ProviderEntry(
                "primary", PROVIDER_CLOUDFLARE, "@cf/zai-org/glm-4.7-flash",
                entry_model, "prompt",
            )
        ],
    )
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert result.confidence == 0.87
    events = [e for e in gateway.last_events if e["outcome"] == "success"]
    assert events and events[0]["provider"] == PROVIDER_CLOUDFLARE
    assert events[0]["duration_ms"] >= 0
