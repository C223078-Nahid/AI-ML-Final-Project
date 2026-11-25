"""Centralized model gateway (concept C2).

Provides one consistent interface for chat and structured-output calls,
backed by an ORDERED, configurable chain of providers (follow-up task 4,
section 5).  Default chain:

    1. Cloudflare Workers AI - GLM-4.7-Flash (active primary)
    2. OpenRouter - NVIDIA Nemotron 3 Ultra (first fallback, attempted
       when the primary hits a rate limit / quota / timeout / transient
       failure)
    3. Anthropic - Claude Haiku 5.5 (the pre-existing fallback, always kept
       as the final resort when its key is configured)

Provider selection and failover order (``LLM_PRIMARY_PROVIDER`` and
``LLM_FALLBACK_PROVIDERS``) are configuration-only changes; every provider
goes through this single gateway - there is no independent model
invocation path.

Structured-output compatibility:
- Providers are tagged with a structured-output mode.  ``native`` uses
  provider-side schema enforcement (``model.with_structured_output``) -
  used by OpenAI and Anthropic.  ``prompt`` uses explicit JSON instructions
  in the prompt, robust client-side JSON parsing (handles Markdown code
  fences and prose around the JSON), Pydantic validation against the
  existing schemas, and one bounded correction attempt - used by OpenRouter
  and, by default, by Cloudflare's OpenAI-compatible endpoint.  Invalid
  structured output is never accepted; persistent failures raise
  ``ModelOutputError`` and follow the normal fallback policy.

Failure policy:
- Fallback-eligible failures: transient provider errors (timeouts,
  connection failures, retryable server errors), rate limits and quota
  exhaustion, authentication failures on a non-final provider, and
  malformed structured output.  Each provider is attempted at most once -
  the chain length is the retry bound, so no call can loop forever.
- NOT fallback-eligible: invalid requests (application bugs such as a bad
  schema), unknown model identifiers (reported with provider-specific
  configuration hints, never silently substituted), and application-level
  validation errors.  These raise immediately with a clear message.

Execution consistency on fallback: the same message list (prompt,
conversation context, retrieved RAG passages) and the same tool definitions
are handed to every attempt, so switching providers never loses context.

Each gateway call is traced by LangSmith (when enabled) with a meaningful
run name and metadata recording the provider, model, attempt, attempt
duration, fallback reason, and structured mode.  No credentials are ever
included in metadata or traces.
"""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Provider registry and structured-output capability modes
# ---------------------------------------------------------------------------
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_CLOUDFLARE = "cloudflare"

#: How structured output is obtained from a provider.
#: - "native": provider-side schema enforcement via
#:   ``model.with_structured_output(schema)``.
#: - "prompt": prompt-instructed JSON + client-side parsing/validation
#:   (for endpoints without ``response_format`` support).
STRUCTURED_NATIVE = "native"
STRUCTURED_PROMPT = "prompt"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_CLOUDFLARE_BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_CLOUDFLARE_MODEL = "@cf/zai-org/glm-4.7-flash"


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------
class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    AUTH = "authentication"
    INVALID_REQUEST = "invalid_request"
    MODEL_OUTPUT = "model_output"
    APP_VALIDATION = "app_validation"
    UNKNOWN = "unknown"


FALLBACK_ELIGIBLE: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.AUTH,
        ErrorCategory.MODEL_OUTPUT,
    }
)


class ModelOutputError(Exception):
    """The model returned a malformed or incomplete response."""


class ModelGatewayError(Exception):
    """A model call failed and no eligible fallback succeeded."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = attempts or []


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Map an exception to an ErrorCategory to decide fallback eligibility."""
    import anthropic
    import openai

    if isinstance(exc, ModelOutputError):
        return ErrorCategory.MODEL_OUTPUT
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return ErrorCategory.RATE_LIMIT
    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ),
    ):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return ErrorCategory.AUTH
    if isinstance(
        exc,
        (
            openai.PermissionDeniedError,
            anthropic.PermissionDeniedError,
        ),
    ):
        return ErrorCategory.AUTH
    if isinstance(
        exc,
        (
            openai.BadRequestError,
            openai.NotFoundError,
            anthropic.BadRequestError,
            anthropic.NotFoundError,
        ),
    ):
        # Includes unknown model identifiers: reported, never substituted.
        return ErrorCategory.INVALID_REQUEST
    if isinstance(exc, ValidationError):
        return ErrorCategory.APP_VALIDATION
    return ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Prompt-based structured output (for providers without response_format)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_MAX_BALANCED_CANDIDATES = 20


def _balanced_json_candidates(text: str) -> list[str]:
    """Return balanced ``{...}`` / ``[...]`` substrings, outermost first.

    Used to recover a JSON object that the model embedded in surrounding
    prose despite the instructions.
    """
    candidates: list[str] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        while start != -1 and len(candidates) < _MAX_BALANCED_CANDIDATES:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : index + 1])
                        break
            start = text.find(start_char, start + 1)
    return candidates


def parse_model_json(text: str) -> Any:
    """Parse the JSON payload out of a raw model reply.

    Tries, in order: the whole reply, every Markdown code-fence body
    (```` ```json ... ``` ```` or plain fences), then balanced JSON objects
    found inside the text.  Raises ``ValueError`` when nothing parses, so
    callers can treat it as a model-output failure.
    """
    if text is None:
        raise ValueError("the model reply was empty")
    stripped = text.strip()
    if not stripped:
        raise ValueError("the model reply was empty")

    candidates: list[str] = [stripped]
    for match in _FENCE_RE.finditer(stripped):
        candidates.append(match.group(1).strip())
    candidates.extend(_balanced_json_candidates(stripped))

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("no valid JSON found in the model reply")


def _message_text(response: Any) -> str:
    """Extract the text of a model response (handles string and block content)."""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _summarize_validation_error(exc: ValidationError) -> str:
    """A compact, non-sensitive summary of a Pydantic validation error."""
    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(piece) for piece in error.get("loc", ()))
        parts.append(f"{location or '(root)'}: {error.get('msg', 'invalid value')}")
    return "; ".join(parts)


def json_contract(schema: type[BaseModel]) -> str:
    """The strict JSON output instruction appended for prompt-mode providers."""
    schema_json = json.dumps(
        schema.model_json_schema(), indent=2, ensure_ascii=False
    )
    return (
        "OUTPUT FORMAT REQUIREMENTS (strict):\n"
        "1. Your entire reply must be a single JSON object. No explanations, "
        "no Markdown code fences, and no text before or after the JSON.\n"
        "2. The JSON object must conform to this JSON Schema:\n"
        f"{schema_json}\n"
        "3. Use the exact enum values listed in the schema. Format dates as "
        "YYYY-MM-DD strings. Use null for optional values you omit. Do not "
        "invent fields that are not in the schema."
    )


class PromptStructuredRunnable:
    """Structured output for models without native schema enforcement.

    Strategy (compatible with endpoints that reject ``response_format``):
    1. Append an explicit JSON contract (the Pydantic-derived JSON Schema)
       to the conversation.
    2. Parse the reply with a robust JSON extractor (handles Markdown code
       fences and JSON embedded in prose).
    3. Validate with the existing Pydantic schema - invalid output is never
       accepted.
    4. On a parse or validation failure, make ONE bounded correction
       attempt, feeding the exact problem back to the model.
    5. If the correction also fails, raise ``ModelOutputError`` (a
       fallback-eligible category) with a clear description.

    When ``PKTA_DEV_CAPTURE_DIR`` is set, the exact final payload (messages
    + JSON contract, including correction turns) is captured immediately
    before each provider call - the development-time proof that retrieved
    passages reach the real model request.
    """

    def __init__(
        self,
        model: Any,
        schema: type[BaseModel],
        *,
        max_corrections: int = 1,
        provider: str = "unknown",
        model_name: str = "unknown",
    ) -> None:
        self._model = model
        self._schema = schema
        self._max_corrections = max_corrections
        self._contract = json_contract(schema)
        self._provider = provider
        self._model_name = model_name

    # -- public API -----------------------------------------------------------
    def invoke(
        self, messages: Sequence[Any], config: dict[str, Any] | None = None
    ) -> BaseModel:
        conversation = list(messages)
        problems: list[str] = []
        last_reply = ""

        for attempt in range(self._max_corrections + 1):
            payload = conversation + [SystemMessage(content=self._contract)]
            if attempt > 0:
                payload = payload + [
                    AIMessage(content=last_reply),
                    HumanMessage(
                        content=(
                            "Your previous reply was not valid. Problem: "
                            f"{problems[-1]}\n"
                            "Respond again with a single JSON object that "
                            "satisfies the schema - no Markdown code fences, "
                            "no text outside the JSON."
                        )
                    ),
                ]
            capture_model_messages(
                "structured",
                self._schema.__name__ + ("_correction" if attempt else ""),
                self._provider,
                self._model_name,
                payload,
            )
            response = self._model.invoke(
                payload, config=self._config_for(config, attempt)
            )

            if getattr(response, "tool_calls", None):
                raise ModelOutputError(
                    "Model attempted a tool call during a structured-output "
                    "request; expected a JSON object."
                )
            last_reply = _message_text(response)
            try:
                parsed = parse_model_json(last_reply)
            except ValueError as exc:
                problems.append(f"reply was not parseable JSON ({exc})")
                continue
            try:
                return self._schema.model_validate(parsed)
            except ValidationError as exc:
                problems.append(
                    "schema validation failed ("
                    + _summarize_validation_error(exc)
                    + ")"
                )
                continue

        raise ModelOutputError(
            f"Model did not produce valid {self._schema.__name__} JSON after "
            f"{self._max_corrections + 1} attempt(s): "
            + "; ".join(problems)
        )

    # -- internals --------------------------------------------------------------
    def _config_for(
        self, config: dict[str, Any] | None, attempt: int
    ) -> dict[str, Any] | None:
        """Label the correction attempt in the tracing config."""
        if attempt == 0 or not config:
            return config
        adjusted = dict(config)
        if adjusted.get("run_name"):
            adjusted["run_name"] = f"{adjusted['run_name']}_correction"
        metadata = dict(adjusted.get("metadata") or {})
        metadata["correction_attempt"] = attempt
        adjusted["metadata"] = metadata
        return adjusted


# ---------------------------------------------------------------------------
# Development-only message capture (follow-up task, section 2.3)
# ---------------------------------------------------------------------------
# "Context included in LLM request: Yes" diagnostics prove the messages were
# BUILT; this capture proves what was actually SENT. When the environment
# variable PKTA_DEV_CAPTURE_DIR points at a directory, every chat and
# structured call appends one JSONL record with the full final message list
# (roles + contents) exactly as handed to the provider runnable.
#
# This is a development and testing tool: it captures document content, so
# it MUST stay disabled (unset) in production. Never enabled by default.

_CAPTURE_ENV = "PKTA_DEV_CAPTURE_DIR"


def _capture_enabled() -> "Path | None":
    import os

    raw = (os.environ.get(_CAPTURE_ENV) or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:  # noqa: BLE001 - capture must never break a call
        return None


def capture_model_messages(
    kind: str,
    run_name: str,
    provider: str,
    model: str,
    messages: Sequence[Any],
) -> None:
    """Write the final message list of one gateway call to a JSONL file.

    Records ``{"ts", "kind", "run_name", "provider", "model", "messages":
    [{"role", "chars", "content"}]}``. Safe to call from tests with
    ``PKTA_DEV_CAPTURE_DIR`` set; a no-op when the variable is unset.
    """
    import datetime as _dt
    import json as _json

    directory = _capture_enabled()
    if directory is None:
        return
    try:
        record = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "run_name": run_name,
            "provider": provider,
            "model": model,
            "messages": [
                {
                    "role": type(m).__name__.replace("Message", "").lower(),
                    "chars": len(str(getattr(m, "content", "") or "")),
                    "content": str(getattr(m, "content", "") or ""),
                }
                for m in messages
            ],
        }
        with open(directory / "model_messages.jsonl", "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - capture must never break a call
        pass


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------
class ProviderEntry:
    """One configured provider slot in the ordered model chain.

    ``role`` is the attempt label used in events and LangSmith metadata
    ("primary", "fallback_1", ...).  ``model`` is None when the provider is
    not configured (missing credentials); such entries are reported as
    skipped, never silently dropped.
    """

    __slots__ = ("role", "provider", "name", "model", "structured_mode")

    def __init__(
        self,
        role: str,
        provider: str,
        name: str,
        model: Any | None,
        structured_mode: str = STRUCTURED_NATIVE,
    ) -> None:
        self.role = role
        self.provider = provider
        self.name = name
        self.model = model
        self.structured_mode = structured_mode

    @property
    def available(self) -> bool:
        return self.model is not None

    def describe(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.name,
            "available": self.available,
            "structured_output": self.structured_mode,
        }


class ModelGateway:
    """Single entry point for all language-model calls.

    The gateway owns an ORDERED chain of providers (follow-up task 4,
    section 5): the primary provider first, then the configurable fallback
    chain (``LLM_FALLBACK_PROVIDERS``), then the legacy Anthropic fallback
    when configured.  Each provider is attempted exactly once - the chain
    length is the retry bound, so no call can loop forever.  Example
    default chain:

        cloudflare (GLM-4.7-Flash, primary)
          -> openrouter (Nemotron 3 Ultra, on rate limit / quota / timeout)
          -> anthropic (Claude Haiku 5.5, legacy fallback, final resort)

    Execution consistency on fallback: the SAME message list (prompt,
    conversation context, retrieved document passages) and the SAME tool
    definitions are handed to every attempt - nothing is rebuilt or lost
    when switching providers.
    """

    def __init__(
        self,
        primary: Any | None,
        fallback: Any | None,
        primary_name: str,
        fallback_name: str,
        *,
        primary_provider: str = PROVIDER_OPENAI,
        fallback_provider: str = PROVIDER_ANTHROPIC,
        primary_structured_mode: str = STRUCTURED_NATIVE,
        fallback_structured_mode: str = STRUCTURED_NATIVE,
        chain: list[ProviderEntry] | None = None,
    ) -> None:
        if chain is None:
            # Backward-compatible two-slot construction (used by the
            # existing tests): primary first, then one fallback.
            chain = [
                ProviderEntry(
                    "primary",
                    primary_provider,
                    primary_name,
                    primary,
                    primary_structured_mode,
                ),
                ProviderEntry(
                    "fallback",
                    fallback_provider,
                    fallback_name,
                    fallback,
                    fallback_structured_mode,
                ),
            ]
        self._chain = chain
        self._structured_cache: dict[tuple[int, type], Runnable] = {}
        # Bookkeeping of the most recent call, for the UI and for tests.
        self.last_events: list[dict[str, Any]] = []

    # -- legacy attribute access (tests, UI) -----------------------------------
    @property
    def _primary(self) -> Any | None:
        return self._chain[0].model if self._chain else None

    @property
    def _fallback(self) -> Any | None:
        """The legacy Anthropic fallback entry (final resort), if any."""
        for entry in self._chain[1:]:
            if entry.provider == PROVIDER_ANTHROPIC:
                return entry.model
        return None

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_settings(cls, settings) -> "ModelGateway":
        """Build the ordered provider chain from application settings.

        The active primary provider is chosen by ``LLM_PRIMARY_PROVIDER``:
        ``cloudflare`` (GLM-4.7-Flash, the current primary), ``openrouter``
        (NVIDIA Nemotron 3 Ultra, the first fallback by default), or
        ``openai`` (the original primary).  Then the ordered fallback chain
        from ``LLM_FALLBACK_PROVIDERS`` is appended, and finally the legacy
        Anthropic fallback (always preserved when its API key is
        configured).  Providers without credentials are kept in the chain
        as unavailable slots so the UI can report them.
        """
        provider = (
            getattr(settings, "llm_primary_provider", None) or PROVIDER_OPENAI
        ).strip().lower()

        chain: list[ProviderEntry] = []

        def _add(role: str, provider_id: str, name: str, model: Any | None,
                 mode: str) -> None:
            chain.append(
                ProviderEntry(role, provider_id, name, model, mode)
            )

        # ---- primary ---------------------------------------------------------
        if provider == PROVIDER_OPENROUTER:
            primary_name = (
                getattr(settings, "openrouter_model", None)
                or DEFAULT_OPENROUTER_MODEL
            )
            # Verified against the OpenRouter model catalog: the free
            # NVIDIA Nemotron 3 Ultra endpoint supports tool calling but
            # NOT the native response_format / JSON-Schema parameter, so
            # structured output is prompt-based for this provider.
            primary_mode = STRUCTURED_PROMPT
            openrouter_key = getattr(settings, "openrouter_api_key", None)
            openrouter_enabled = getattr(
                settings, "openrouter_enabled", True
            )
            base_url = (
                getattr(settings, "openrouter_base_url", None)
                or DEFAULT_OPENROUTER_BASE_URL
            ).rstrip("/")
            primary_model: Any | None = None
            if openrouter_key and openrouter_enabled:
                from langchain_openai import ChatOpenAI

                primary_model = ChatOpenAI(
                    model=primary_name,
                    api_key=openrouter_key,
                    base_url=base_url,
                    # OpenRouter serves the OpenAI-compatible
                    # chat-completions API; the Responses API must stay off.
                    use_responses_api=False,
                    temperature=0.2,
                    max_tokens=settings.max_output_tokens,
                    timeout=settings.model_timeout_seconds,
                    max_retries=0,  # the gateway owns the retry policy
                    # Recommended OpenRouter attribution header (optional).
                    default_headers={
                        "X-Title": "personal-knowledge-task-assistant"
                    },
                )
            _add("primary", PROVIDER_OPENROUTER, primary_name,
                 primary_model, primary_mode)
        elif provider == PROVIDER_CLOUDFLARE:
            primary_model = _build_cloudflare_model(settings)
            _add(
                "primary",
                PROVIDER_CLOUDFLARE,
                _cloudflare_model_name(settings),
                primary_model,
                _cloudflare_structured_mode(settings),
            )
        elif provider == PROVIDER_OPENAI:
            if getattr(settings, "openai_api_key", None):
                from langchain_openai import ChatOpenAI

                primary_model = ChatOpenAI(
                    model=settings.primary_model,
                    api_key=settings.openai_api_key,
                    # GPT-6 tool calling and structured outputs are fully
                    # supported on the Responses API.
                    use_responses_api=True,
                    temperature=None,
                    max_tokens=settings.max_output_tokens,
                    timeout=settings.model_timeout_seconds,
                    max_retries=0,  # the gateway owns the retry policy
                )
            else:
                primary_model = None
            _add("primary", PROVIDER_OPENAI, settings.primary_model,
                 primary_model, STRUCTURED_NATIVE)
        else:
            # Unknown provider: reported by Settings.validate at startup.
            # Never silently substituted with another provider here.
            _add("primary", provider, provider, None, STRUCTURED_NATIVE)

        # ---- ordered fallback chain (LLM_FALLBACK_PROVIDERS) -----------------
        raw_fallbacks = getattr(settings, "llm_fallback_providers", "") or ""
        seen: set[str] = {provider}
        index = 0
        for entry in raw_fallbacks.split(","):
            fallback_id = entry.strip().lower()
            if not fallback_id or fallback_id in seen:
                continue
            seen.add(fallback_id)
            index += 1
            role = f"fallback_{index}"
            if fallback_id == PROVIDER_CLOUDFLARE:
                _add(role, PROVIDER_CLOUDFLARE,
                     _cloudflare_model_name(settings),
                     _build_cloudflare_model(settings),
                     _cloudflare_structured_mode(settings))
            elif fallback_id == PROVIDER_ANTHROPIC:
                _add(role, PROVIDER_ANTHROPIC, settings.fallback_model,
                     _build_anthropic_model(settings), STRUCTURED_NATIVE)
            elif fallback_id == PROVIDER_OPENROUTER:
                name = (
                    getattr(settings, "openrouter_model", None)
                    or DEFAULT_OPENROUTER_MODEL
                )
                model = None
                key = getattr(settings, "openrouter_api_key", None)
                openrouter_enabled = getattr(
                    settings, "openrouter_enabled", True
                )
                if key and openrouter_enabled:
                    from langchain_openai import ChatOpenAI

                    model = ChatOpenAI(
                        model=name,
                        api_key=key,
                        base_url=(
                            getattr(settings, "openrouter_base_url", None)
                            or DEFAULT_OPENROUTER_BASE_URL
                        ).rstrip("/"),
                        use_responses_api=False,
                        temperature=0.2,
                        max_tokens=settings.max_output_tokens,
                        timeout=settings.model_timeout_seconds,
                        max_retries=0,
                        default_headers={
                            "X-Title": "personal-knowledge-task-assistant"
                        },
                    )
                _add(role, PROVIDER_OPENROUTER, name, model, STRUCTURED_PROMPT)
            elif fallback_id == PROVIDER_OPENAI:
                model = None
                if getattr(settings, "openai_api_key", None):
                    from langchain_openai import ChatOpenAI

                    model = ChatOpenAI(
                        model=settings.primary_model,
                        api_key=settings.openai_api_key,
                        use_responses_api=True,
                        temperature=None,
                        max_tokens=settings.max_output_tokens,
                        timeout=settings.model_timeout_seconds,
                        max_retries=0,
                    )
                _add(role, PROVIDER_OPENAI, settings.primary_model, model,
                     STRUCTURED_NATIVE)
            else:
                # Unknown entries are reported by Settings.validate; the
                # gateway keeps a visible unavailable slot for them.
                _add(role, fallback_id, fallback_id, None, STRUCTURED_NATIVE)

        # ---- legacy Anthropic fallback: always the final resort ---------------
        if PROVIDER_ANTHROPIC not in seen:
            _add(f"fallback_{index + 1}", PROVIDER_ANTHROPIC,
                 settings.fallback_model, _build_anthropic_model(settings),
                 STRUCTURED_NATIVE)

        return cls(
            primary=None, fallback=None,
            primary_name="", fallback_name="",
            chain=chain,
        )

    # -- introspection -------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Legacy two-slot view (primary + first Anthropic fallback) plus the
        full ordered ``chain``."""
        primary = self._chain[0] if self._chain else None
        legacy_fallback = next(
            (e for e in self._chain[1:] if e.provider == PROVIDER_ANTHROPIC),
            None,
        )
        if legacy_fallback is None and len(self._chain) > 1:
            legacy_fallback = self._chain[1]
        return {
            "primary_model": primary.name if primary else "",
            "primary_provider": primary.provider if primary else "",
            "primary_available": bool(primary and primary.available),
            "primary_structured_output": (
                primary.structured_mode if primary else STRUCTURED_NATIVE
            ),
            "fallback_model": legacy_fallback.name if legacy_fallback else "",
            "fallback_provider": (
                legacy_fallback.provider if legacy_fallback else ""
            ),
            "fallback_available": bool(
                legacy_fallback and legacy_fallback.available
            ),
            "fallback_structured_output": (
                legacy_fallback.structured_mode if legacy_fallback else ""
            ),
            "chain": [entry.describe() for entry in self._chain],
        }

    def _attempts(self) -> list[ProviderEntry]:
        """Configured providers in failover order; unavailable ones are
        logged as skipped."""
        attempts: list[ProviderEntry] = []
        for entry in self._chain:
            if entry.available:
                attempts.append(entry)
            else:
                self.last_events.append(
                    {
                        "attempt": entry.role,
                        "model": entry.name,
                        "provider": entry.provider,
                        "outcome": "skipped",
                        "reason": "not configured (missing credentials)",
                    }
                )
        return attempts

    def _no_model_error(self) -> ModelGatewayError:
        return ModelGatewayError(
            "No language model is configured. Check the API keys in your "
            f".env file (active primary provider: {self._chain[0].provider if self._chain else 'none'}; "
            "expected keys: OPENROUTER_API_KEY for 'openrouter', "
            "CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN for 'cloudflare', "
            "OPENAI_API_KEY for 'openai', ANTHROPIC_API_KEY for the "
            "legacy fallback).",
            ErrorCategory.APP_VALIDATION,
        )

    # -- error enrichment -----------------------------------------------------------
    @staticmethod
    def _enrich_error(entry: ProviderEntry, exc: Exception,
                      category: ErrorCategory) -> str:
        """Provider-specific configuration hints for clear error messages."""
        text = str(exc)
        if entry.provider == PROVIDER_CLOUDFLARE and category in (
            ErrorCategory.INVALID_REQUEST,
            ErrorCategory.AUTH,
        ):
            hint = (
                " [Cloudflare provider hint: verify CLOUDFLARE_MODEL matches "
                "an available Workers AI model (current: "
                f"{entry.name}) and CLOUDFLARE_ACCOUNT_ID is correct; the "
                "token must have Workers AI permissions - see the README "
                "section on Cloudflare credentials.]"
            )
            if "hint" not in text:
                text = f"{text}{hint}"
        return text

    # -- public API ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] | None = None,
        run_name: str = "model_chat",
        metadata: dict[str, Any] | None = None,
    ) -> AIMessage:
        """Plain chat completion, optionally with read-only tools bound.

        The identical message list (prompt, conversation context, retrieved
        document passages) and the identical tool definitions are handed to
        every provider attempt, so switching providers never loses context.
        """
        self.last_events = []
        attempts_log: list[dict[str, Any]] = []
        attempts = self._attempts()
        if not attempts:
            raise self._no_model_error()

        fallback_reason: str | None = None
        for entry in attempts:
            try:
                runnable: Runnable = entry.model
                if tools:
                    runnable = entry.model.bind_tools(list(tools))
                capture_model_messages(
                    "chat", run_name, entry.provider, entry.name,
                    list(messages),
                )
                started = time.perf_counter()
                result = runnable.invoke(
                    list(messages),
                    config={
                        "run_name": run_name,
                        "metadata": {
                            **(metadata or {}),
                            "component": "llm_gateway",
                            "provider": entry.provider,
                            "model": entry.name,
                            "attempt": entry.role,
                            **(
                                {"fallback_reason": fallback_reason}
                                if fallback_reason
                                else {}
                            ),
                        },
                    },
                )
                duration_ms = round((time.perf_counter() - started) * 1000)
                if not isinstance(result, AIMessage):
                    raise ModelOutputError(
                        f"Model {entry.name} returned "
                        f"{type(result).__name__} instead of an AI message."
                    )
                self.last_events.append(
                    {
                        "attempt": entry.role,
                        "model": entry.name,
                        "provider": entry.provider,
                        "outcome": "success",
                        "duration_ms": duration_ms,
                        **(
                            {"fallback_reason": fallback_reason}
                            if fallback_reason
                            else {}
                        ),
                    }
                )
                return result
            except Exception as exc:  # noqa: BLE001 - classified below
                category = classify_exception(exc)
                event = {
                    "attempt": entry.role,
                    "model": entry.name,
                    "provider": entry.provider,
                    "outcome": "failed",
                    "category": category.value,
                    "error": self._enrich_error(entry, exc, category)[:400],
                }
                attempts_log.append(event)
                self.last_events.append(event)
                if category not in FALLBACK_ELIGIBLE:
                    raise ModelGatewayError(
                        f"Model call failed on {entry.role} model "
                        f"'{entry.name}' via {entry.provider} "
                        f"({category.value}): "
                        f"{self._enrich_error(entry, exc, category)}",
                        category,
                        attempts_log,
                    ) from exc
                # eligible failure: continue to the next provider in order
                fallback_reason = f"{entry.provider}_{category.value}"
        raise ModelGatewayError(
            "All configured models failed. Attempted: "
            + "; ".join(
                f"{event['attempt']}({event['provider']}/{event['model']})"
                f": {event['error']}"
                for event in attempts_log
            ),
            ErrorCategory.UNKNOWN,
            attempts_log,
        )

    def structured(
        self,
        schema: type[BaseModel],
        messages: Sequence[Any],
        *,
        run_name: str = "structured_output",
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Structured output call validated against a Pydantic schema.

        The per-provider structured-output mode is selected automatically:
        native schema enforcement where the endpoint supports it, explicit
        prompt instructions plus robust parsing and one bounded correction
        attempt where it does not (OpenRouter's free Nemotron endpoint,
        Cloudflare's OpenAI-compatible endpoint by default).
        """
        self.last_events = []
        attempts_log: list[dict[str, Any]] = []
        attempts = self._attempts()
        if not attempts:
            raise self._no_model_error()

        fallback_reason: str | None = None
        for entry in attempts:
            try:
                runnable = self._structured_runnable(
                    entry.model, schema, entry.structured_mode,
                    entry.provider, entry.name,
                )
                started = time.perf_counter()
                result = runnable.invoke(
                    list(messages),
                    config={
                        "run_name": run_name,
                        "metadata": {
                            **(metadata or {}),
                            "component": "llm_gateway",
                            "provider": entry.provider,
                            "model": entry.name,
                            "attempt": entry.role,
                            "schema": schema.__name__,
                            "structured_mode": entry.structured_mode,
                            **(
                                {"fallback_reason": fallback_reason}
                                if fallback_reason
                                else {}
                            ),
                        },
                    },
                )
                duration_ms = round((time.perf_counter() - started) * 1000)
                if not isinstance(result, schema):
                    raise ModelOutputError(
                        f"Model {entry.name} returned "
                        f"{type(result).__name__} instead of "
                        f"{schema.__name__}."
                    )
                self.last_events.append(
                    {
                        "attempt": entry.role,
                        "model": entry.name,
                        "provider": entry.provider,
                        "outcome": "success",
                        "duration_ms": duration_ms,
                        **(
                            {"fallback_reason": fallback_reason}
                            if fallback_reason
                            else {}
                        ),
                    }
                )
                return result
            except Exception as exc:  # noqa: BLE001 - classified below
                category = classify_exception(exc)
                event = {
                    "attempt": entry.role,
                    "model": entry.name,
                    "provider": entry.provider,
                    "outcome": "failed",
                    "category": category.value,
                    "error": self._enrich_error(entry, exc, category)[:400],
                }
                attempts_log.append(event)
                self.last_events.append(event)
                if category not in FALLBACK_ELIGIBLE:
                    raise ModelGatewayError(
                        f"Structured call failed on {entry.role} model "
                        f"'{entry.name}' via {entry.provider} "
                        f"({category.value}): "
                        f"{self._enrich_error(entry, exc, category)}",
                        category,
                        attempts_log,
                    ) from exc
                fallback_reason = f"{entry.provider}_{category.value}"
        raise ModelGatewayError(
            "All configured models failed. Attempted: "
            + "; ".join(
                f"{event['attempt']}({event['provider']}/{event['model']})"
                f": {event['error']}"
                for event in attempts_log
            ),
            ErrorCategory.UNKNOWN,
            attempts_log,
        )

    # -- internals --------------------------------------------------------------------
    def _structured_runnable(
        self, model: Any, schema: type[BaseModel], mode: str, provider: str,
        model_name: str,
    ) -> Runnable:
        key = (id(model), schema)
        cached = self._structured_cache.get(key)
        if cached is None:
            if mode == STRUCTURED_PROMPT:
                cached = PromptStructuredRunnable(
                    model, schema, provider=provider, model_name=model_name
                )
            else:
                cached = model.with_structured_output(schema)
            self._structured_cache[key] = cached
        return cached


# ---------------------------------------------------------------------------
# Cloudflare Workers AI provider construction (follow-up task, section 6)
# ---------------------------------------------------------------------------
def _cloudflare_model_name(settings) -> str:
    return (
        getattr(settings, "cloudflare_model", None)
        or DEFAULT_CLOUDFLARE_MODEL
    )


def _cloudflare_base_url(settings) -> str:
    """The complete OpenAI-compatible base URL for Cloudflare Workers AI.

    Built as {CLOUDFLARE_BASE_URL}/{CLOUDFLARE_ACCOUNT_ID}/ai/v1 - per the
    task specification and the Cloudflare Workers AI OpenAI-compatibility
    documentation (chat completions live at .../ai/v1/chat/completions and
    the OpenAI SDK accepts the .../ai/v1 prefix as base_url).
    """
    base = (
        getattr(settings, "cloudflare_base_url", None)
        or DEFAULT_CLOUDFLARE_BASE_URL
    ).rstrip("/")
    account = getattr(settings, "cloudflare_account_id", None) or ""
    return f"{base}/{account}/ai/v1"


def _cloudflare_structured_mode(settings) -> str:
    mode = (getattr(settings, "cloudflare_structured_mode", None) or "").strip()
    return mode.lower() if mode in ("prompt", "native") else STRUCTURED_PROMPT


def _build_cloudflare_model(settings) -> Any | None:
    """Build the GLM-4.7-Flash ChatOpenAI client, or None when unconfigured.

    Uses the existing ``langchain-openai`` package against Cloudflare's
    OpenAI-compatible endpoint (verified against the Workers AI docs).  No
    credentials are ever logged or traced - they only flow into the client.
    """
    enabled = getattr(settings, "cloudflare_enabled", True)
    account_id = getattr(settings, "cloudflare_account_id", None)
    api_token = getattr(settings, "cloudflare_api_token", None)
    if not (enabled and account_id and api_token):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_cloudflare_model_name(settings),
        api_key=api_token,
        base_url=_cloudflare_base_url(settings),
        # Cloudflare serves the OpenAI-compatible chat-completions API;
        # the Responses API must stay off.
        use_responses_api=False,
        temperature=0.2,
        max_tokens=getattr(settings, "max_output_tokens", 1500),
        timeout=getattr(settings, "model_timeout_seconds", 90.0),
        max_retries=0,  # the gateway owns the retry policy
    )


def _build_anthropic_model(settings) -> Any | None:
    """Build the legacy Anthropic fallback model, or None when unconfigured."""
    if not getattr(settings, "anthropic_api_key", None):
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.fallback_model,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
        max_tokens=getattr(settings, "max_output_tokens", 1500),
        timeout=getattr(settings, "model_timeout_seconds", 90.0),
        max_retries=0,
    )
