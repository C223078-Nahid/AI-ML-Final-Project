"""Scriptable fakes used by the automated tests (no network, no API keys)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from core.models import GroundedAnswer, IntentResult, TaskProposal


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Build an AIMessage that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@dataclass
class FakeGateway:
    """Mimics the ModelGateway interface with scripted behavior.

    - ``chat(tools=...)`` consumes ``agent_script`` (tool-call responses);
      when the script is empty it returns a plain "DONE" message.
    - ``chat()`` without tools returns ``chat_text``.
    - ``structured(schema, ...)`` returns the scripted schema instance
      (``intent`` / ``proposal`` / ``grounded`` shortcuts, or any schema
      instance placed in ``structured_map`` - used by the fact-check
      workflow tests for ExtractedClaims / ClaimAssessmentBatch /
      FactCheckReport).
    All received messages are recorded in ``seen_messages`` for assertions.
    """

    intent: IntentResult | None = None
    proposal: TaskProposal | None = None
    grounded: GroundedAnswer | None = None
    agent_script: list[AIMessage] = field(default_factory=list)
    chat_text: str = "This is a general answer."
    structured_error: Exception | None = None
    # Fails ONLY the intent-classification call (used to exercise the
    # deterministic heuristic fallback without breaking grounded answers).
    intent_error: Exception | None = None
    chat_error: Exception | None = None
    seen_messages: list[tuple[list[Any], bool]] = field(default_factory=list)
    structured_calls: list[str] = field(default_factory=list)
    chat_calls_with_tools: int = 0
    #: Follow-up task 4: script arbitrary structured-output schemas by
    #: type, e.g. {ExtractedClaims: ExtractedClaims(claims=[...])}.  A
    #: list value is consumed item by item (one per call).
    structured_map: dict = field(default_factory=dict)

    def chat(
        self, messages, tools=None, run_name=None, metadata=None
    ) -> AIMessage:
        self.seen_messages.append((list(messages), tools is not None))
        if self.chat_error is not None:
            raise self.chat_error
        if tools is not None:
            self.chat_calls_with_tools += 1
            if self.agent_script:
                return self.agent_script.pop(0)
            return AIMessage(content="DONE")
        return AIMessage(content=self.chat_text)

    def structured(self, schema, messages, run_name=None, metadata=None):
        self.structured_calls.append(schema.__name__)
        self.seen_messages.append((list(messages), False))
        if self.structured_error is not None:
            raise self.structured_error
        if schema is IntentResult:
            if self.intent_error is not None:
                raise self.intent_error
            if self.intent is None:
                raise AssertionError("FakeGateway: intent not scripted")
            return self.intent
        if schema is TaskProposal:
            if self.proposal is None:
                raise AssertionError("FakeGateway: proposal not scripted")
            return self.proposal
        if schema is GroundedAnswer:
            if self.grounded is None:
                raise AssertionError("FakeGateway: grounded answer not scripted")
            return self.grounded
        if schema in self.structured_map:
            scripted = self.structured_map[schema]
            if isinstance(scripted, list):
                if not scripted:
                    raise AssertionError(
                        f"FakeGateway: script for {schema.__name__} exhausted"
                    )
                return scripted.pop(0)
            return scripted
        raise AssertionError(f"FakeGateway: unexpected schema {schema.__name__}")


class Raw:
    """Wrap a value that a ScriptedModel should return verbatim (used to
    simulate malformed model output, e.g. a plain string instead of an
    AIMessage)."""

    def __init__(self, value: Any) -> None:
        self.value = value


class ScriptedModel:
    """Duck-typed chat model for ModelGateway unit tests.

    Behavior can be: an exception to raise, a callable, a Raw wrapper, an
    AIMessage, or a plain string (wrapped into an AIMessage). Implements
    invoke / bind_tools / with_structured_output. For structured calls use
    ``structured_result`` (a schema instance, a Raw value, or an exception).
    """

    def __init__(self, behavior: Any, structured_result: Any = None) -> None:
        self.behavior = behavior
        self.structured_result = structured_result
        self.calls = 0
        self.structured_calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        behavior = self.behavior
        if isinstance(behavior, BaseException):
            raise behavior
        if isinstance(behavior, Raw):
            return behavior.value
        if callable(behavior):
            return behavior(self, messages)
        if isinstance(behavior, AIMessage):
            return behavior
        return AIMessage(content=str(behavior))

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        outer = self

        class _Runnable:
            def invoke(self, messages, config=None):
                outer.structured_calls += 1
                result = outer.structured_result
                if isinstance(result, BaseException):
                    raise result
                if isinstance(result, Raw):
                    return result.value
                return result

        return _Runnable()


class SequentialTextModel:
    """Chat model fake that replays scripted replies in order.

    Used to exercise the prompt-based structured-output path of the model
    gateway (OpenRouter / NVIDIA Nemotron 3 Ultra): the gateway sends the
    messages plus a JSON contract, and this fake replies with the next
    scripted text (or raises it, when the item is an exception).  Every
    received message list is recorded in ``received`` so tests can assert on
    the JSON contract and correction turns.

    ``with_structured_output`` fails the test if called, because prompt-mode
    providers must never use native structured output.
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.received: list[list[Any]] = []
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        self.received.append(list(messages))
        if not self.replies:
            raise AssertionError("SequentialTextModel: no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, AIMessage):
            return reply
        return AIMessage(content=str(reply))

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        raise AssertionError(
            "native structured output must not be used for prompt-mode "
            "providers (no response_format support)"
        )


def make_openai_error(error_class, status: int):
    """Construct a real openai SDK error (httpx2-based) for tests."""
    import httpx2
    import openai

    assert issubclass(error_class, openai.APIError)
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(status, request=request)
    return error_class("simulated failure", response=response, body=None)


def make_anthropic_error(error_class, status: int):
    """Construct a real anthropic SDK error (httpx2-based) for tests."""
    import anthropic
    import httpx2

    assert issubclass(error_class, anthropic.APIError)
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return error_class("simulated failure", response=response, body=None)
