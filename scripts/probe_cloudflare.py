"""Cloudflare Workers AI (GLM-4.7-Flash) integration probe.

Verifies the ACTUAL endpoint and model against the task specification:

    POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/chat/completions
    model: @cf/zai-org/glm-4.7-flash

Levels (each runs only when the previous one succeeded / when the needed
credentials are present):

  0. Endpoint reachability  - no credentials: an unauthenticated request
     must return a structured 4xx (proves the endpoint exists and speaks
     the OpenAI-compatible error format; a DNS/network failure instead
     proves the sandbox cannot reach Cloudflare).
  1. Plain chat completion  - needs CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN.
  2. Tool calling           - a weather-tool request; the model must emit
     a function call (GLM-4.7-Flash supports tool calling per its model
     page).
  3. Streaming              - SSE chunks must arrive and concatenate.
  4. Structured output      - prompt-mode JSON + Pydantic validation
     (exactly what the application does by default).

Run:
    python3 scripts/probe_cloudflare.py            # level 0 only
    CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \
        python3 scripts/probe_cloudflare.py        # all levels

Credentials are read from the environment / .env only; they are never
printed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/zai-org/glm-4.7-flash")
BASE_PREFIX = (
    os.getenv("CLOUDFLARE_BASE_URL", "https://api.cloudflare.com/client/v4/accounts")
    or "https://api.cloudflare.com/client/v4/accounts"
).rstrip("/")
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {name}: {detail}")


def chat_url() -> str:
    return f"{BASE_PREFIX}/{ACCOUNT_ID}/ai/v1/chat/completions"


def _post(payload: dict, stream: bool = False, timeout: float = 60.0):
    import httpx

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
    }
    return httpx.post(
        chat_url(), json=payload, headers=headers, timeout=timeout,
        stream=stream,
    )


def level0_reachability() -> bool:
    """Unauthenticated request against the real endpoint: expect a
    structured 4xx (401/403/400/404), never a network error."""
    import httpx

    url = f"{BASE_PREFIX}/00000000000000000000000000000000/ai/v1/chat/completions"
    try:
        response = httpx.post(
            url,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer invalid-probe-token",
                     "Content-Type": "application/json"},
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        record(
            "endpoint-reachability", False,
            f"could not reach api.cloudflare.com from this environment "
            f"({type(exc).__name__}: {str(exc)[:120]})",
        )
        return False
    detail = f"HTTP {response.status_code} from the real endpoint"
    if 400 <= response.status_code < 500:
        body = response.text[:160].replace("\n", " ")
        record("endpoint-reachability", True, f"{detail} - {body}")
        return True
    record("endpoint-reachability", False, detail + " (unexpected status)")
    return False


def level1_chat() -> bool:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": "Which company develops GLM models?"},
        ],
        "max_tokens": 120,
    }
    try:
        response = _post(payload)
    except Exception as exc:  # noqa: BLE001
        record("chat-completion", False, f"{type(exc).__name__}: {str(exc)[:160]}")
        return False
    if response.status_code != 200:
        record(
            "chat-completion", False,
            f"HTTP {response.status_code}: {response.text[:200]}",
        )
        return False
    body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
    except Exception:  # noqa: BLE001
        record("chat-completion", False, f"unexpected body shape: {str(body)[:200]}")
        return False
    record(
        "chat-completion", True,
        f"model={body.get('model', MODEL)} reply={content[:80]!r} "
        f"usage={usage}",
    )
    return True


def level2_tool_calling() -> bool:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo? Use the tool."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "max_tokens": 200,
    }
    try:
        response = _post(payload)
    except Exception as exc:  # noqa: BLE001
        record("tool-calling", False, f"{type(exc).__name__}: {str(exc)[:160]}")
        return False
    if response.status_code != 200:
        record(
            "tool-calling", False,
            f"HTTP {response.status_code}: {response.text[:200]}",
        )
        return False
    message = response.json()["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        record(
            "tool-calling", False,
            f"no tool call in the reply: {str(message)[:160]}",
        )
        return False
    call = tool_calls[0]["function"]
    record(
        "tool-calling", True,
        f"function={call['name']} args={call.get('arguments')}",
    )
    return True


def level3_streaming() -> bool:
    import httpx

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from 1 to 5 slowly."}],
        "max_tokens": 80,
        "stream": True,
    }
    try:
        chunks: list[str] = []
        with _post(payload, stream=True) as response:
            if response.status_code != 200:
                record(
                    "streaming", False,
                    f"HTTP {response.status_code}: {response.text[:160]}",
                )
                return False
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except Exception:  # noqa: BLE001
                    continue
                content = delta.get("content")
                if content:
                    chunks.append(content)
        text = "".join(chunks)
        if not text:
            record("streaming", False, "no content deltas arrived")
            return False
        record("streaming", True, f"{len(chunks)} chunks -> {text[:70]!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        record("streaming", False, f"{type(exc).__name__}: {str(exc)[:160]}")
        return False


def level4_structured_output() -> bool:
    """Prompt-mode structured output - exactly what the app's gateway does
    for the Cloudflare provider (JSON contract + Pydantic validation)."""
    from pydantic import BaseModel

    class WeatherAnswer(BaseModel):
        city: str
        temperature_c: float
        summary: str

    schema = json.dumps(WeatherAnswer.model_json_schema(), indent=2)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Tokyo is 22.5 degrees and sunny.\n\n"
                    "OUTPUT FORMAT REQUIREMENTS (strict):\n"
                    "1. Your entire reply must be a single JSON object.\n"
                    f"2. It must conform to this JSON Schema:\n{schema}\n"
                    "3. No Markdown fences, no text outside the JSON."
                ),
            }
        ],
        "max_tokens": 200,
    }
    try:
        response = _post(payload)
    except Exception as exc:  # noqa: BLE001
        record("structured-output", False, f"{type(exc).__name__}: {str(exc)[:160]}")
        return False
    if response.status_code != 200:
        record(
            "structured-output", False,
            f"HTTP {response.status_code}: {response.text[:160]}",
        )
        return False
    raw = response.json()["choices"][0]["message"]["content"]
    # Reuse the application's robust parser (handles fences/prose).
    from core.llm_gateway import parse_model_json

    try:
        parsed = WeatherAnswer.model_validate(parse_model_json(raw))
    except Exception as exc:  # noqa: BLE001
        record(
            "structured-output", False,
            f"reply not schema-valid: {str(exc)[:120]} reply={raw[:120]!r}",
        )
        return False
    record(
        "structured-output", True,
        f"parsed {parsed.model_dump()}",
    )
    return True


def main() -> int:
    print(f"Cloudflare Workers AI probe - model {MODEL}")
    print(f"endpoint base: {chat_url() if ACCOUNT_ID else '(account id not set)'}")
    print()
    reachable = level0_reachability()
    if not (ACCOUNT_ID and TOKEN):
        print(
            "\nCredentials not set - stopping after the reachability probe. "
            "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env to "
            "run the live model tests (levels 1-4)."
        )
        return 0 if reachable else 1

    ok = level1_chat()
    if ok:
        level2_tool_calling()
        level3_streaming()
        level4_structured_output()
    failures = [name for name, passed, _ in RESULTS if not passed]
    print()
    if failures:
        print(f"FAILED levels: {', '.join(failures)}")
        return 1
    print("All probe levels passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
