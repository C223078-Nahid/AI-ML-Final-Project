"""C2: model gateway, fallback policy, and structured-output contract."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from core.llm_gateway import (
    ErrorCategory,
    FALLBACK_ELIGIBLE,
    ModelGateway,
    ModelGatewayError,
    ModelOutputError,
    classify_exception,
)
from core.models import Intent, IntentResult, TaskProposal

from tests.fakes import (
    Raw,
    ScriptedModel,
    make_anthropic_error,
    make_openai_error,
)

import anthropic
import openai


MESSAGES = [HumanMessage(content="hello")]


def _gateway(primary, fallback):
    return ModelGateway(
        primary=primary, fallback=fallback,
        primary_name="gpt-6-luna", fallback_name="claude-haiku-5-5",
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exception,expected",
    [
        (make_openai_error(openai.RateLimitError, 429), ErrorCategory.RATE_LIMIT),
        (make_anthropic_error(anthropic.RateLimitError, 429), ErrorCategory.RATE_LIMIT),
        (make_openai_error(openai.AuthenticationError, 401), ErrorCategory.AUTH),
        (make_openai_error(openai.InternalServerError, 500), ErrorCategory.TRANSIENT),
        (make_openai_error(openai.BadRequestError, 400), ErrorCategory.INVALID_REQUEST),
        (make_openai_error(openai.NotFoundError, 404), ErrorCategory.INVALID_REQUEST),
        (ModelOutputError("bad output"), ErrorCategory.MODEL_OUTPUT),
        (RuntimeError("whatever"), ErrorCategory.UNKNOWN),
    ],
)
def test_classify_exception(exception, expected):
    assert classify_exception(exception) is expected


def test_only_expected_categories_are_fallback_eligible():
    assert FALLBACK_ELIGIBLE == {
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.AUTH,
        ErrorCategory.MODEL_OUTPUT,
    }


# ---------------------------------------------------------------------------
# Chat fallback behavior
# ---------------------------------------------------------------------------
def test_primary_success_uses_no_fallback():
    primary = ScriptedModel("primary answer")
    fallback = ScriptedModel("fallback answer")
    result = _gateway(primary, fallback).chat(MESSAGES)
    assert result.content == "primary answer"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "primary_error",
    [
        make_openai_error(openai.RateLimitError, 429),
        make_openai_error(openai.AuthenticationError, 401),
        make_openai_error(openai.InternalServerError, 500),
        make_openai_error(openai.PermissionDeniedError, 403),
    ],
)
def test_eligible_primary_failure_falls_back(primary_error):
    primary = ScriptedModel(primary_error)
    fallback = ScriptedModel("fallback answer")
    gateway = _gateway(primary, fallback)
    result = gateway.chat(MESSAGES)
    assert result.content == "fallback answer"
    assert primary.calls == 1 and fallback.calls == 1
    assert gateway.last_events[0]["outcome"] == "failed"
    assert gateway.last_events[1]["outcome"] == "success"


@pytest.mark.parametrize(
    "primary_error",
    [
        make_openai_error(openai.BadRequestError, 400),
        make_openai_error(openai.NotFoundError, 404),  # unknown model id
    ],
)
def test_invalid_request_does_not_fall_back(primary_error):
    primary = ScriptedModel(primary_error)
    fallback = ScriptedModel("should not be used")
    gateway = _gateway(primary, fallback)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)
    assert excinfo.value.category is ErrorCategory.INVALID_REQUEST
    assert fallback.calls == 0


def test_transient_connection_error_falls_back():
    request = __import__("httpx2").Request("POST", "https://api.openai.com")
    primary_error = openai.APIConnectionError(message="boom", request=request)
    primary = ScriptedModel(primary_error)
    fallback = ScriptedModel("fallback answer")
    result = _gateway(primary, fallback).chat(MESSAGES)
    assert result.content == "fallback answer"


def test_both_providers_unavailable_raises():
    error = make_openai_error(openai.RateLimitError, 429)
    primary = ScriptedModel(error)
    fallback = ScriptedModel(make_anthropic_error(anthropic.RateLimitError, 429))
    gateway = _gateway(primary, fallback)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)
    assert len(excinfo.value.attempts) == 2


def test_no_models_configured_raises():
    gateway = _gateway(None, None)
    with pytest.raises(ModelGatewayError):
        gateway.chat(MESSAGES)


def test_primary_unavailable_goes_straight_to_fallback():
    fallback = ScriptedModel("fallback answer")
    gateway = ModelGateway(
        primary=None, fallback=fallback,
        primary_name="gpt-6-luna", fallback_name="claude-haiku-5-5",
    )
    result = gateway.chat(MESSAGES)
    assert result.content == "fallback answer"
    assert any(event["outcome"] == "skipped" for event in gateway.last_events)


def test_non_ai_message_output_counts_as_model_output_failure():
    primary = ScriptedModel(Raw("just a string, not an AIMessage"))
    fallback = ScriptedModel(AIMessage(content="proper answer"))
    gateway = _gateway(primary, fallback)
    result = gateway.chat(MESSAGES)
    assert result.content == "proper answer"
    assert gateway.last_events[0]["category"] == "model_output"


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------
def test_structured_success_primary():
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    primary = ScriptedModel(None, structured_result=good)
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert primary.structured_calls == 1
    assert fallback.structured_calls == 0


def test_malformed_model_response_falls_back():
    good = IntentResult(intent=Intent.GENERAL_CONVERSATION, confidence=0.9)
    primary = ScriptedModel(None, structured_result="not a schema instance")
    fallback = ScriptedModel(None, structured_result=good)
    gateway = _gateway(primary, fallback)
    result = gateway.structured(IntentResult, MESSAGES)
    assert isinstance(result, IntentResult)
    assert primary.structured_calls == 1
    assert fallback.structured_calls == 1


def test_structured_invalid_request_raises_without_fallback():
    error = make_openai_error(openai.BadRequestError, 400)
    primary = ScriptedModel(None, structured_result=error)
    fallback = ScriptedModel(None, structured_result="unused")
    gateway = _gateway(primary, fallback)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.structured(IntentResult, MESSAGES)
    assert excinfo.value.category is ErrorCategory.INVALID_REQUEST
    assert fallback.structured_calls == 0


def test_app_validation_error_never_triggers_fallback():
    from pydantic import ValidationError

    validation_error: ValidationError
    try:
        IntentResult(intent="bogus")
    except ValidationError as caught:
        validation_error = caught
    else:
        raise AssertionError("expected ValidationError")
    primary = ScriptedModel(validation_error)
    fallback = ScriptedModel("unused")
    gateway = _gateway(primary, fallback)
    with pytest.raises(ModelGatewayError) as excinfo:
        gateway.chat(MESSAGES)
    assert excinfo.value.category is ErrorCategory.APP_VALIDATION
    assert fallback.calls == 0


# ---------------------------------------------------------------------------
# Construction from settings
# ---------------------------------------------------------------------------
def test_from_settings_builds_both_models():
    class _Settings:
        openai_api_key = "sk-test"
        anthropic_api_key = "sk-ant-test"
        primary_model = "gpt-6-luna"
        fallback_model = "claude-haiku-5-5"
        max_output_tokens = 1500
        model_timeout_seconds = 30.0

    gateway = ModelGateway.from_settings(_Settings())
    info = gateway.describe()
    assert info["primary_available"] and info["fallback_available"]
    assert info["primary_model"] == "gpt-6-luna"
    assert info["fallback_model"] == "claude-haiku-5-5"


def test_from_settings_without_keys_disables_models():
    class _Settings:
        openai_api_key = None
        anthropic_api_key = None
        primary_model = "gpt-6-luna"
        fallback_model = "claude-haiku-5-5"
        max_output_tokens = 1500
        model_timeout_seconds = 30.0

    gateway = ModelGateway.from_settings(_Settings())
    info = gateway.describe()
    assert not info["primary_available"]
    assert not info["fallback_available"]
    with pytest.raises(ModelGatewayError):
        gateway.chat(MESSAGES)


def test_task_proposal_used_through_gateway_validates():
    # Sanity check that TaskProposal works as a structured schema target.
    proposal = TaskProposal(
        title="Write tests", confirmation_message="ok"
    )
    primary = ScriptedModel(None, structured_result=proposal)
    gateway = _gateway(primary, None)
    result = gateway.structured(TaskProposal, MESSAGES)
    assert result.title == "Write tests"
