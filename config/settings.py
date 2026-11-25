"""Application configuration.

All configurable values live here and can be overridden through environment
variables (see ``.env.example``) or a local ``.env`` file.  Nothing in this
module ever prints secret values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Canonical provider identifiers (mirrored by core/llm_gateway.py, which
#: owns the provider implementations).  Kept as plain strings here so this
#: lightweight config module never imports the LangChain stack.
PROVIDER_ANTHROPIC = "anthropic"
KNOWN_PROVIDERS = ("openai", "openrouter", "cloudflare", "anthropic")


class Settings(BaseSettings):
    """Application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Language models ---------------------------------------------------
    # LLM_PRIMARY_PROVIDER selects which configured provider serves as the
    # active primary model.  Follow-up task 4, section 5: Cloudflare Workers
    # AI GLM-4.7-Flash is the PRIMARY; OpenRouter (NVIDIA Nemotron 3 Ultra)
    # is the FIRST fallback; the remaining providers (openai, and the legacy
    # anthropic fallback) stay available further down the chain.  Switching
    # the order is a configuration-only change.
    llm_primary_provider: str = "cloudflare"
    # LLM_FALLBACK_PROVIDERS is the ordered fallback chain attempted when
    # the primary provider fails with a fallback-eligible error (rate limit,
    # quota, timeout, transient unavailability, retryable server error,
    # connection failure).  Comma-separated.  Unknown entries are rejected
    # at startup.  The legacy Anthropic fallback is ALWAYS appended at the
    # end of the chain when configured and not already listed, so the
    # pre-existing failover behaviour is preserved.
    llm_fallback_providers: str = "openrouter"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    primary_model: str = "gpt-6-luna"
    fallback_model: str = "claude-haiku-5-5"

    # --- OpenRouter (first fallback provider) ------------------------------
    # NVIDIA: Nemotron 3 Ultra (free) via the OpenRouter chat-completions
    # endpoint.  OPENROUTER_ENABLED=false removes the provider from the
    # chain entirely.  The free endpoint does not support the native OpenAI
    # ``response_format`` JSON-Schema parameter, so structured output is
    # produced through explicit prompt instructions (see core/llm_gateway).
    openrouter_enabled: bool = True
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"

    # --- Cloudflare Workers AI (primary provider) ---------------------------
    # GLM-4.7-Flash (Zhipu AI) through Cloudflare's OpenAI-compatible
    # endpoint.  The full chat-completions URL is constructed as
    # {CLOUDFLARE_BASE_URL}/{CLOUDFLARE_ACCOUNT_ID}/ai/v1 - the account ID
    # and API token live ONLY in the local .env file, never in source code.
    # CLOUDFLARE_ENABLED=false removes the provider from the chain entirely.
    # CLOUDFLARE_STRUCTURED_MODE: "prompt" (JSON contract + client-side
    # Pydantic validation, the safe default for the OpenAI-compatible
    # endpoint) or "native" (provider-side response_format enforcement).
    cloudflare_enabled: bool = True
    cloudflare_account_id: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_model: str = "@cf/zai-org/glm-4.7-flash"
    cloudflare_base_url: str = "https://api.cloudflare.com/client/v4/accounts"
    cloudflare_structured_mode: str = "prompt"

    # --- Model-driven function calling (follow-up task 4, section 6) -------
    # LLM_TOOL_CALLING_ENABLED=false binds no tools to the model (direct
    # answers only).  LLM_MAX_TOOL_ITERATIONS bounds the agent tool loop;
    # LLM_MAX_TOTAL_TOOL_CALLS is the absolute per-request tool budget.
    llm_tool_calling_enabled: bool = True

    # --- Web search (follow-up task 4, section 9) ---------------------------
    # Optional model-invoked web_search tool.  Provider-independent
    # service (core/web_search.py); "tavily" is the initial provider.
    # Disabled / gracefully unavailable when the key is not configured.
    web_search_enabled: bool = True
    web_search_provider: str = "tavily"
    tavily_api_key: str | None = None
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 15.0

    # --- Document fact-checking (follow-up task 4, section 8) ---------------
    # Bounds of the explicit fact-check workflow: never an expensive
    # full-document pass on ordinary document questions.
    fact_check_max_claims: int = 8
    fact_check_max_web_queries: int = 4

    @property
    def cloudflare_available(self) -> bool:
        """Whether the Cloudflare provider is fully configured."""
        return bool(
            self.cloudflare_enabled
            and self.cloudflare_account_id
            and self.cloudflare_api_token
        )

    @property
    def cloudflare_chat_base_url(self) -> str:
        """The complete OpenAI-compatible base URL for Cloudflare Workers AI.

        Built from CLOUDFLARE_BASE_URL + CLOUDFLARE_ACCOUNT_ID + /ai/v1, per
        the task specification and the Cloudflare Workers AI documentation.
        """
        base = (self.cloudflare_base_url or "").rstrip("/")
        account = self.cloudflare_account_id or ""
        return f"{base}/{account}/ai/v1"

    @property
    def web_search_available(self) -> bool:
        """Whether the web-search tool is fully configured (follow-up
        task 4, section 9: gracefully unavailable without an API key -
        never an error)."""
        if not self.web_search_enabled:
            return False
        if self.web_search_provider == "tavily":
            return bool(self.tavily_api_key)
        return False

    def fallback_provider_chain(self) -> list[str]:
        """The ordered, de-duplicated fallback chain from
        LLM_FALLBACK_PROVIDERS, with the legacy Anthropic fallback appended
        when configured and not already present."""
        raw = (self.llm_fallback_providers or "").strip()
        chain: list[str] = []
        for entry in raw.split(","):
            provider = entry.strip().lower()
            if provider and provider not in chain:
                chain.append(provider)
        if (
            self.anthropic_api_key
            and PROVIDER_ANTHROPIC not in chain
        ):
            chain.append(PROVIDER_ANTHROPIC)
        return chain

    # --- LangSmith -----------------------------------------------------------
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "personal-knowledge-task-assistant"

    # --- Embeddings and local storage ----------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_dir: str = "./data"
    qdrant_path: str = "./data/qdrant"
    sqlite_path: str = "./data/app.db"
    checkpoint_db_path: str = "./data/checkpoints.db"

    # --- Limits and tuning ----------------------------------------------------
    max_file_size_mb: int = 20
    max_documents: int = 10
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.30
    max_message_chars: int = 4000
    max_output_tokens: int = 1500
    max_tool_iterations: int = 6
    max_total_tool_calls: int = 12
    model_timeout_seconds: float = 90.0
    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 100
    max_context_chars: int = 12000
    max_history_messages: int = 12

    # --- Application timezone (current date/time questions) -------------------
    # Used by the deterministic datetime utility (follow-up task, section
    # 5.2 C): "What day is today?" must be answered from the application
    # clock in this zone, never from model training data.
    app_timezone: str = "Asia/Dhaka"

    # --- Derived helpers -------------------------------------------------------
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create every local storage directory used by the application."""
        for raw_path in (self.data_dir, self.qdrant_path):
            Path(raw_path).mkdir(parents=True, exist_ok=True)
        for raw_path in (self.sqlite_path, self.checkpoint_db_path):
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)

    def validate(self) -> tuple[list[str], list[str]]:
        """Validate configuration at startup.

        Returns ``(errors, warnings)``.  Errors mean the app cannot serve
        model calls; warnings mean a reduced feature set.  Secret values are
        never included in the returned messages.
        """
        errors: list[str] = []
        warnings: list[str] = []

        provider = (self.llm_primary_provider or "").strip().lower()
        if provider not in ("openai", "openrouter", "cloudflare"):
            errors.append(
                "LLM_PRIMARY_PROVIDER must be 'openai', 'openrouter', or "
                f"'cloudflare' (got '{self.llm_primary_provider}')."
            )
        elif provider == "openrouter":
            if not self.openrouter_api_key:
                errors.append(
                    "LLM_PRIMARY_PROVIDER is 'openrouter' but "
                    "OPENROUTER_API_KEY is not set. Add the key to .env or "
                    "switch back with LLM_PRIMARY_PROVIDER=openai."
                )
            if self.openai_api_key:
                warnings.append(
                    "Informational: the original primary model "
                    f"({self.primary_model}) is configured but not active. "
                    "Set LLM_PRIMARY_PROVIDER=openai to restore it."
                )
        elif provider == "cloudflare":
            if not self.cloudflare_enabled:
                errors.append(
                    "LLM_PRIMARY_PROVIDER is 'cloudflare' but "
                    "CLOUDFLARE_ENABLED is false. Set CLOUDFLARE_ENABLED=true."
                )
            elif not (self.cloudflare_account_id and self.cloudflare_api_token):
                missing = []
                if not self.cloudflare_account_id:
                    missing.append("CLOUDFLARE_ACCOUNT_ID")
                if not self.cloudflare_api_token:
                    missing.append("CLOUDFLARE_API_TOKEN")
                errors.append(
                    "LLM_PRIMARY_PROVIDER is 'cloudflare' but "
                    + " and ".join(missing)
                    + " (is|are) not set. Add them to .env - see the README "
                    "section on Cloudflare credentials."
                )
            if self.openai_api_key:
                warnings.append(
                    "Informational: the original primary model "
                    f"({self.primary_model}) is configured but not active. "
                    "Set LLM_PRIMARY_PROVIDER=openai to restore it."
                )
        else:
            if not (
                self.openai_api_key
                or self.anthropic_api_key
                or self.cloudflare_available
            ):
                errors.append(
                    "No language model API key is configured. Copy .env.example "
                    "to .env and set OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or "
                    "the Cloudflare credentials."
                )
            elif not self.openai_api_key:
                warnings.append(
                    "OPENAI_API_KEY is not set: the primary model "
                    f"({self.primary_model}) is disabled and requests will "
                    f"go directly to the fallback chain "
                    f"({', '.join(self.fallback_provider_chain()) or 'none'})."
                )

        # --- Cloudflare fallback configuration ---------------------------------
        chain = self.fallback_provider_chain()
        for entry in chain:
            if entry not in KNOWN_PROVIDERS:
                errors.append(
                    f"Unknown provider '{entry}' in LLM_FALLBACK_PROVIDERS. "
                    f"Known providers: {', '.join(KNOWN_PROVIDERS)}."
                )
        chain_valid = all(entry in KNOWN_PROVIDERS for entry in chain)
        if chain_valid and "cloudflare" in chain and provider != "cloudflare":
            if not self.cloudflare_enabled:
                warnings.append(
                    "CLOUDFLARE_ENABLED is false: the Cloudflare provider is "
                    "skipped in the fallback chain."
                )
            elif not (self.cloudflare_account_id and self.cloudflare_api_token):
                missing = []
                if not self.cloudflare_account_id:
                    missing.append("CLOUDFLARE_ACCOUNT_ID")
                if not self.cloudflare_api_token:
                    missing.append("CLOUDFLARE_API_TOKEN")
                warnings.append(
                    "Cloudflare is in the fallback chain (LLM_FALLBACK_PROVIDERS) "
                    "but " + " and ".join(missing) + " (is|are) not set, so it "
                    "will be skipped. Add them to .env to enable the Cloudflare "
                    "GLM-4.7-Flash fallback - see the README."
                )
        if chain_valid and provider != "openrouter" and "openrouter" in chain \
                and not self.openrouter_api_key:
            warnings.append(
                "'openrouter' appears in LLM_FALLBACK_PROVIDERS but "
                "OPENROUTER_API_KEY is not set; that entry will be skipped."
            )
        if chain_valid and provider != "openai" and "openai" in chain \
                and not self.openai_api_key:
            warnings.append(
                "'openai' appears in LLM_FALLBACK_PROVIDERS but "
                "OPENAI_API_KEY is not set; that entry will be skipped."
            )

        # The legacy Anthropic fallback stays available (appended to the
        # chain automatically when its key is configured).
        if not self.anthropic_api_key and PROVIDER_ANTHROPIC not in chain:
            warnings.append(
                "ANTHROPIC_API_KEY is not set: the legacy fallback model "
                f"({self.fallback_model}) is disabled. If every configured "
                "provider fails, requests will fail with a descriptive error."
            )
        if self.cloudflare_structured_mode not in ("prompt", "native"):
            errors.append(
                "CLOUDFLARE_STRUCTURED_MODE must be 'prompt' or 'native' "
                f"(got '{self.cloudflare_structured_mode}')."
            )

        # --- Web search (follow-up task 4, section 9) -------------------------
        if self.web_search_provider not in ("tavily",):
            errors.append(
                "WEB_SEARCH_PROVIDER must be 'tavily' "
                f"(got '{self.web_search_provider}')."
            )
        elif self.web_search_enabled and not self.tavily_api_key:
            warnings.append(
                "WEB_SEARCH_ENABLED is true but TAVILY_API_KEY is not set: "
                "the web_search tool stays unavailable (ordinary chat and "
                "document questions are unaffected). Get a key at "
                "https://app.tavily.com - see the README section on web "
                "search."
            )
        if self.web_search_max_results < 1 or self.web_search_max_results > 10:
            errors.append("WEB_SEARCH_MAX_RESULTS must be between 1 and 10.")
        if self.web_search_timeout_seconds < 3:
            errors.append("WEB_SEARCH_TIMEOUT must be at least 3 seconds.")
        if self.fact_check_max_claims < 1:
            errors.append("FACT_CHECK_MAX_CLAIMS must be at least 1.")

        if not self.primary_model or not self.fallback_model:
            errors.append("PRIMARY_MODEL and FALLBACK_MODEL must both be set.")
        if not self.openrouter_model:
            errors.append("OPENROUTER_MODEL must be set.")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            errors.append(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS."
            )
        if self.retrieval_top_k < 1:
            errors.append("RETRIEVAL_TOP_K must be at least 1.")
        if self.max_documents < 1:
            errors.append("MAX_DOCUMENTS must be at least 1.")
        if self.max_tool_iterations < 1:
            errors.append("MAX_TOOL_ITERATIONS must be at least 1.")
        # The timezone must be a valid IANA zone name: the deterministic
        # datetime utility depends on it at question time.
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(self.app_timezone)
        except Exception:
            errors.append(
                f"APP_TIMEZONE '{self.app_timezone}' is not a valid IANA"
                " timezone identifier (for example 'Asia/Dhaka')."
            )
        return errors, warnings

    @property
    def active_primary_model(self) -> str:
        """The model identifier of the currently selected primary provider."""
        provider = (self.llm_primary_provider or "").strip().lower()
        if provider == "openrouter":
            return self.openrouter_model
        if provider == "cloudflare":
            return self.cloudflare_model
        return self.primary_model

    @property
    def active_primary_available(self) -> bool:
        """Whether the selected primary provider has its API key configured."""
        provider = (self.llm_primary_provider or "").strip().lower()
        if provider == "openrouter":
            return bool(self.openrouter_api_key)
        if provider == "cloudflare":
            return self.cloudflare_available
        return bool(self.openai_api_key)

    def public_summary(self) -> dict[str, object]:
        """A non-sensitive snapshot of the configuration for the Settings UI."""
        return {
            "llm_primary_provider": (
                self.llm_primary_provider or ""
            ).strip().lower(),
            "primary_model": self.active_primary_model,
            "openai_primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "primary_available": self.active_primary_available,
            "fallback_available": bool(self.anthropic_api_key),
            "fallback_chain": self.fallback_provider_chain(),
            "cloudflare_enabled": self.cloudflare_enabled,
            "cloudflare_available": self.cloudflare_available,
            "cloudflare_model": self.cloudflare_model,
            "openrouter_base_url": self.openrouter_base_url,
            "openrouter_model": self.openrouter_model,
            "llm_tool_calling_enabled": self.llm_tool_calling_enabled,
            "web_search_enabled": self.web_search_enabled,
            "web_search_provider": self.web_search_provider,
            "web_search_available": self.web_search_available,
            "embedding_model": self.embedding_model,
            "langsmith_tracing": self.langsmith_tracing,
            "langsmith_project": self.langsmith_project,
            "data_dir": self.data_dir,
            "sqlite_path": self.sqlite_path,
            "checkpoint_db_path": self.checkpoint_db_path,
            "qdrant_path": self.qdrant_path,
            "retrieval_top_k": self.retrieval_top_k,
            "max_documents": self.max_documents,
            "max_file_size_mb": self.max_file_size_mb,
            "app_timezone": self.app_timezone,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings instance (cached)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests that mutate the environment)."""
    get_settings.cache_clear()
