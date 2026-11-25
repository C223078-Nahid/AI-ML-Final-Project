# Personal Knowledge & Task Assistant

A local, single-user AI workspace that lets a student manage personal
documents, ask questions with verifiable citations, and create tasks through
conversation — with explicit human approval before anything is written.

Built with Python, Streamlit, LangChain, LangGraph, Qdrant (local mode), and
SQLite. Everything runs on your machine; the only external calls are to the
language-model APIs you configure.

---

## 1. Project overview

The assistant is an academic project demonstrating practical AI engineering:

- **C1** — Prompts and structured output (Pydantic-validated).
- **C2** — LangChain model layer with a fallback model and LangSmith tracing.
- **C3** — Tools and a bounded agent loop.
- **C4** — Conversation memory via LangGraph persistence.
- **C5** — Safety guardrails and human approval.
- **C6** — RAG with verifiable citations.
- **C7** — LangGraph workflow orchestration.
- **E2** — Durable pause/resume with `interrupt()`.
- **E3** — Long-term memory (explicitly saved preferences).

A single developer can understand, maintain, and demonstrate the entire
codebase.

## 2. Features

- **ChatGPT-inspired interface** — dark theme, left sidebar in the exact
  required hierarchy (application name at the top-left, **New Chat**,
  Documents, Tasks, Settings, Approvals, then the **Chats** history —
  visually separated and independently scrollable), everything
  left-aligned with no taglines.
- **Sidebar toggle on the sidebar side** — the ☰ control sits at the
  top-left of the sidebar next to the brand; when the sidebar is collapsed,
  a reopen control floats at the left edge of the application (the same
  top-left corner in both states). Toggling never resets the active
  conversation and never interrupts pending approvals or workflows
  (tooltips: "Close sidebar" / "Open sidebar").
- **Multiple chat sessions** — create new chats, switch between them, and
  delete conversations (with confirmation). Each chat keeps its own isolated
  context via a distinct LangGraph `thread_id`; titles are derived from the
  first message (no extra model call).
- **Shared document library** — ONE global Qdrant collection serves every
  chat. Upload a document once and ask about it in any conversation; only
  conversation *histories* are per-thread, document *access* is global.
- **Composer attachments (ChatGPT-style)** — a **+** control on the left of
  the message input opens the file browser directly (one click, no
  popover); selected files are ingested immediately through the shared
  pipeline and appear as preview chips **above** the input (file-type
  badge, filename, indexing status, remove). **Drag and drop** a supported
  document anywhere on the chat area to upload it the same way.
- **Request cancellation** — while the assistant works, Send is replaced
  by a **Stop** button; Stop cancels further model calls, tool
  executions, and task writes at the next safe point, marks the response
  as stopped, and re-enables the input (see section 17.5).
- **Real progress phases** — the thinking indicator shows the actual
  application phase ("Searching uploaded documents...", "Retrieving
  relevant passages...", "Preparing response...") reported by the
  workflow itself — never fabricated reasoning.
- **Message processing state** — a submitted message runs in a background
  worker; every request carries a unique execution token and a dedup
  fingerprint, and the state resets on success, on errors, *and* on
  cancellation.
- **Fast ingestion** — duplicate uploads are detected by content hash and
  skipped instantly; the embedding model is loaded once and warmed up in the
  background at startup; Qdrant upserts are batched; Streamlit reruns never
  reprocess an already-indexed file. Typical timings: a small TXT in
  ~0.03 s, a 25-page PDF in ~0.1 s (after the one-time model load).
- **Document library management** — per-document status, chunk and page
  counts, removal with confirmation; re-uploading a changed file under the
  same name replaces the old version instead of duplicating it; a startup
  **reconciliation** removes orphan Qdrant points and stale full-text rows
  so the UI chunk counts always match the actual index.
- **Verbatim small-document chunking** — a section that fits the chunk
  budget is indexed exactly as uploaded (labels like `Name:` and paragraph
  breaks survive); only oversized sections fall back to sentence-split
  chunks with overlap.
- **Conversational document Q&A (RAG), retrieval-first with hybrid
  retrieval** — the application itself executes retrieval for document
  questions (the model is never the one deciding whether to search), using
  **Qdrant semantic search plus a SQLite FTS5 keyword fallback**: exact
  facts ("What is my name?") are found even when embedding similarity is
  marginal, results are deduplicated, and contextual follow-ups such as
  *"What was the reason for choosing it?"* are resolved into standalone
  search queries. Answers are grounded in the retrieved passages with
  citations (filename, page or section, supporting passage, chunk id,
  score).
- **Citation correction, not discarding** — passages carry short ordinal
  source ids (S1, S2, ...) that free endpoints echo reliably; a citation
  that references the right passage through the filename or quoted text is
  *corrected* to the real chunk, and a flawed first reply (unjustified
  abstention or unresolvable citations) gets ONE corrective retry before
  the app ever shows an abstention.
- **Honest abstention vs. distinct conditions** — the assistant abstains
  only when the retrieved evidence genuinely lacks the answer (a phone
  number that is not in the document stays unanswered); an empty library,
  a retrieval-infrastructure failure, and missing evidence are three
  different, distinguishable responses.
- **Seven-way query routing** — document questions (RAG), general-knowledge
  questions (answered directly from model knowledge, never pretending to
  have searched), current date/time questions (the MODEL invokes the
  registered `get_current_datetime` tool; the application clock in the
  configured `APP_TIMEZONE` stays the data source — never training data,
  never document abstention, and a fabricated timestamp can never survive),
  explicit fact-check requests, task listing, task creation, and
  unsupported requests. Deterministic safety nets keep each category on
  its path even when the (free-endpoint) classifier mislabels it.
- **Model-driven function calling (follow-up task 4)** — the model decides
  when to call a registered tool through a centralized tool registry
  (search documents, read a document, list tasks, current date/time,
  web search, restricted page reading, claim verification).  Every tool
  argument is validated against its schema independently of the model,
  execution is bounded (iteration and call budgets), repeated identical
  calls are suppressed, and the RESTRICTED `create_task` tool is never
  even offered to the model (see section 9.3).
- **Optional web search as an AI tool** — when a Tavily key is configured,
  the model can search the public web for current or independent
  information; results are untrusted evidence with real URLs only, and
  page reading is restricted to search-result URLs with full SSRF
  protection (see section 9.4).
- **Document fact-checking** — explicit verification requests extract the
  document's verifiable claims, compare them with the model's knowledge,
  optionally gather independent web evidence, and produce a structured
  report separating what the document says from what is actually
  supported (nine verification categories, never a forced true/false;
  invented citations are stripped by application-side validation — see
  section 9.5).
- **Clickable suggested prompts (follow-up task 4)** — the new-chat
  suggestions are centered and clickable: clicking inserts the prompt into
  the composer draft (shared state), ready to edit and send; it never
  auto-submits, never overwrites an existing draft, and never creates a
  chat.
- **Safe RAG diagnostics** — every ingestion and retrieval logs a
  `[rag-diag]` line (collection name, storage path, indexed points,
  retrieved chunk count, context length, provider/model, routing decision,
  citation-validation outcome, final answer outcome) with **no document
  content and no credentials**, so indexing, retrieval, context delivery,
  and routing can be verified from the terminal. `PKTA_DEV_CAPTURE_DIR`
  (development only) additionally captures the exact final message list
  before every model call.
- **Task management** — list and filter tasks (open / completed /
  cancelled) with title, description, due date, priority, status, and
  creation date.
- **Task proposals with human approval** — asking "add a task ..." produces
  a validated proposal; nothing is saved until you click **Approve** in the
  chat or the Approval Center. Rejection discards the proposal. Pending
  approvals survive restarts and even the deletion of their conversation.
- **Conversation memory** — each chat thread keeps its own context across
  messages, page refreshes, and application restarts.
- **Long-term memory** — preferences you explicitly save on the Settings
  page are injected into every new conversation, in every thread.
- **Ordered model fallback** — Cloudflare Workers AI GLM-4.7-Flash
  (primary) -> OpenRouter NVIDIA Nemotron 3 Ultra (first fallback) -> the
  preserved OpenAI and legacy Anthropic providers; on eligible failures
  (rate limit, quota, timeout, transient) the next provider receives the
  same request, conversation, retrieved passages, and tool definitions
  (see section 9.1).
- **LangSmith tracing** — optional; the app works fine without it.
- **Persistence** — tasks, preferences, documents, conversations, pending
  approvals, and full workflow state live in local SQLite files; chat
  history and the document index survive application restarts and browser
  refreshes.

## 3. Technology stack

| Layer | Technology | Pinned version |
|---|---|---|
| Language | Python | 3.11+ (verified on 3.12) |
| UI | Streamlit | 1.67.0 |
| LLM orchestration | LangChain | 1.5.2 |
| Workflow orchestration | LangGraph | 1.2.24 |
| Observability | LangSmith | 0.15.1 |
| Primary model (active, temporary) | `nvidia/nemotron-3-ultra-550b-a55b:free` via LangChain OpenAI on OpenRouter | — |
| Primary model (original, preserved) | `gpt-6-luna` via LangChain OpenAI (Responses API) | — |
| Fallback model | `claude-haiku-5-5` via LangChain Anthropic | — |
| Structured output | Pydantic | 2.12.5 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | 6.3.0 |
| Vector database | Qdrant, local persistent mode | 1.20.0 |
| App database / checkpointing | SQLite (+ `langgraph-checkpoint-sqlite`) | — |
| PDF / DOCX parsing | pypdf / python-docx | 6.6.0 / 1.2.0 |
| Testing | pytest | 9.0.2 |
| Environment | python-dotenv | 1.2.2 |

All model identifiers were verified available (November 2026; the
OpenRouter catalog was queried live for the Nemotron entry and its supported
parameters). If a provider retires a model, set a different id in `.env` —
the gateway reports unknown-model errors instead of silently substituting.

## 4. Architecture explanation

The LangGraph `StateGraph` is the application's actual controller:

```
START
  |
input_guardrail  --(refusal / invalid)--> final_response
  |
intent_router  --(unsupported)----------> final_response
  |
  +-- document_question --> document_retrieval --> grounded_answer --> final_response
  |                            (hybrid: Qdrant + FTS5; planner only when
  |                             direct retrieval is empty)
  +-- general_knowledge --> general_response -----------------------> final_response
  +-- general_conversation --> general_response --------------------> final_response
  +-- current_datetime --> datetime_response -----------------------> final_response
  |                          (deterministic application clock, APP_TIMEZONE;
  |                           datetime safety net overrides misroutes)
  +-- list_tasks --------> task_listing ----------------------------> final_response
  +-- create_task -------> task_proposal --> human_approval (interrupt)
  |                                            +-- approved --> task_execution --> final_response
  |                                            +-- rejected -------------------> final_response
  |
final_response --> END
```

Routing safety nets (deterministic, in `core/guardrails.py` +
`graph/nodes.py`): a time-shaped question never enters the RAG branch
(the reported "What day is today?" abstention); a document-shaped question
mislabeled as general conversation / general knowledge / unsupported is
pulled back to the document branch when the library has documents; a
contextual follow-up stays on the branch of its thread; and when the
classifier call itself fails, a datetime-first deterministic classifier
routes the request.

- **Model gateway** (`core/llm_gateway.py`) — one interface for chat and
  structured output; tries the primary model, falls back to Claude Haiku 5.5
  on eligible failures (rate limit, transient, auth, malformed output).
- **Tools** — `search_documents` and `list_tasks` are read-only and bound to
  models; `create_task` is a restricted write tool invoked only by the
  `task_execution` node after approval, never by a model.
- **Persistence** — `app.db` (tasks, preferences, documents, conversations,
  pending operations) and `checkpoints.db` (LangGraph state) are separate
  SQLite files.
- **Citation trust** — citations are validated against the chunks that were
  actually retrieved; invalid ones are dropped and metadata is overwritten
  with the authoritative chunk metadata.

See `docs/architecture.md` for the full node-by-node description and
`docs/demo_guide.md` for a scripted demonstration of every concept.

## 5. Project folder structure

```
personal-knowledge-task-assistant/
├── app.py                  # Streamlit entry point
├── config/settings.py      # env-driven configuration + validation
├── core/                   # models (Pydantic), prompts, gateway, guardrails
├── graph/                  # state, nodes, routing, workflow, persistence
├── tools/                  # search_documents, list_tasks, create_task, memory
├── rag/                    # parsers, chunking, embeddings, vector store, retrieval, ingestion
├── database/               # connection, schema, tasks, memory, documents, conversations
├── ui/                     # chat, documents, tasks, approvals, settings,
│                          #   theme (dark CSS), uploads (shared processing), graph service
├── tests/                  # pytest suite (230+ tests, fully offline)
├── samples/                # synthetic demo corpus (5 documents)
├── scripts/                # make_sample_docs.py, seed_demo.py,
│                          #   verify_openrouter.py, verify_pipeline.py
├── docs/                   # architecture.md, demo_guide.md
├── data/                   # created at runtime (git-ignored)
├── .streamlit/             # dark theme configuration
├── .env.example
├── requirements.txt
└── README.md
```

## 6. Prerequisites

- Python **3.11 or newer** (3.12 verified).
- An **OpenRouter API key** — this is the currently active primary provider
  (temporary setup, free tier; see section 9.1).
- An **OpenAI API key** (the original primary provider — keep it configured
  so you can switch back at any time) and/or an **Anthropic API key**
  (fallback model).
- Optional: a free **LangSmith** account for tracing.
- Internet access for the first run only (the embedding model, ~90 MB,
  downloads from Hugging Face once, then runs fully locally).

## 7. Installation instructions

```bash
cd personal-knowledge-task-assistant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (smaller download, CPU-only PyTorch, no CUDA):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 8. Virtual environment setup

The commands above already create and activate a `.venv`. If you prefer
`uv` or `conda`, any Python 3.11+ environment works — only `pip install -r
requirements.txt` matters. Deactivate with `deactivate` when done.

## 9. Environment variable configuration

```bash
cp .env.example .env
```

Then edit `.env`:

- `LLM_PRIMARY_PROVIDER` — selects the active primary provider:
  `cloudflare` (current primary: GLM-4.7-Flash), `openrouter` (NVIDIA
  Nemotron 3 Ultra), or `openai` (the original setup).  See sections
  9.1 / 9.2 below.
- `LLM_FALLBACK_PROVIDERS` — the ordered fallback chain (default
  `openrouter`, the first fallback); the legacy Anthropic fallback is
  always appended at the end when configured.
- `CLOUDFLARE_ENABLED` / `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` /
  `CLOUDFLARE_MODEL` / `CLOUDFLARE_BASE_URL` / `CLOUDFLARE_STRUCTURED_MODE`
  — the Cloudflare Workers AI **primary** provider (GLM-4.7-Flash); see
  section 9.2.
- `OPENROUTER_API_KEY` — key for the OpenRouter **first fallback**
  (`OPENROUTER_ENABLED=false` removes it from the chain).
- `LLM_TOOL_CALLING_ENABLED` / `LLM_MAX_TOOL_ITERATIONS` /
  `LLM_MAX_TOTAL_TOOL_CALLS` — model-driven function calling (see
  section 9.3).
- `WEB_SEARCH_ENABLED` / `WEB_SEARCH_PROVIDER` / `TAVILY_API_KEY` /
  `WEB_SEARCH_MAX_RESULTS` / `WEB_SEARCH_TIMEOUT` — the optional
  model-invoked web-search tool (see section 9.4).
- `FACT_CHECK_MAX_CLAIMS` / `FACT_CHECK_MAX_WEB_QUERIES` — bounds of the
  document fact-checking workflow (see section 9.5).
- `APP_TIMEZONE` — the timezone of the `get_current_datetime` tool
  (follow-up task 4: the model decides when to call it).
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — original primary and fallback
  keys (never commit `.env`; it is git-ignored).
- `OPENROUTER_MODEL` (default `nvidia/nemotron-3-ultra-550b-a55b:free`) /
  `PRIMARY_MODEL` (default `gpt-6-luna`) / `FALLBACK_MODEL` (default
  `claude-haiku-5-5`).
- `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` — optional
  tracing.
- Storage paths (`DATA_DIR`, `QDRANT_PATH`, `SQLITE_PATH`,
  `CHECKPOINT_DB_PATH`) and limits (`MAX_FILE_SIZE_MB`, `MAX_DOCUMENTS`,
  `RETRIEVAL_TOP_K`, `CHUNK_SIZE_TOKENS`, ...) with sane defaults.

Configuration is validated at startup; the sidebar shows clear errors (e.g.
missing keys) and warnings (e.g. fallback disabled) without printing secret
values.

### 9.1 Model providers: Cloudflare primary, OpenRouter first fallback

The provider priority (follow-up task 4, section 5):

| Priority | Provider | Model |
|---|---|---|
| Primary | Cloudflare Workers AI | `@cf/zai-org/glm-4.7-flash` (GLM-4.7-Flash) |
| Fallback 1 | OpenRouter | `nvidia/nemotron-3-ultra-550b-a55b:free` (NVIDIA Nemotron 3 Ultra) |
| Fallback 2+ | Existing providers | OpenAI (GPT-6 Luna) and the legacy Anthropic fallback (Claude Haiku 5.5) — preserved and always appended when configured |

Switching the order is a configuration-only change
(`LLM_PRIMARY_PROVIDER` / `LLM_FALLBACK_PROVIDERS`); every provider still
goes through the single centralized model gateway.  When the primary fails
with a rate limit, quota exhaustion, timeout, or transient error, the next
provider receives the SAME messages (conversation context, retrieved
passages, tool definitions, and any in-flight tool-call exchange) — a
fallback never loses the tool-call context.

**OpenRouter (first fallback) model identifier:**

```text
nvidia/nemotron-3-ultra-550b-a55b:free
```

served through the OpenRouter chat-completions endpoint
`https://openrouter.ai/api/v1/chat/completions` via the existing
`langchain-openai` `ChatOpenAI` integration (no new dependencies).

**OpenRouter account and API-key setup:**

1. Create a free account at <https://openrouter.ai/>.
2. Obtain an API key on the Keys page: <https://openrouter.ai/keys>.
3. In `.env`, set `OPENROUTER_API_KEY=<your key>` (leave the other provider
   keys configured so the fallback and the switch-back keep working).

**Required environment variables for this setup** (already in
`.env.example`):

```env
LLM_PRIMARY_PROVIDER=openrouter
OPENROUTER_API_KEY=<your key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

**How to switch back to the original model provider** — change one value in
`.env` and restart the app; no code changes are required:

```env
LLM_PRIMARY_PROVIDER=openai
```

`OPENAI_API_KEY` and `PRIMARY_MODEL=gpt-6-luna` keep working unchanged, and
the Anthropic fallback (`claude-haiku-5-5`) stays available in both
configurations. The Settings page always shows the active provider and
model, and every request records which provider handled it (LangSmith
metadata and the gateway event log).

**Known compatibility considerations:**

- The free Nemotron 3 Ultra endpoint supports **tool calling** (`tools`,
  `tool_choice`) but **not** the native OpenAI `response_format` /
  JSON-Schema parameter (verified against the live OpenRouter model catalog,
  November 2026). Structured output (`IntentResult`, `TaskProposal`,
  `GroundedAnswer`) therefore uses explicit prompt instructions with the
  JSON Schema embedded, a robust JSON parser (handles Markdown code fences
  and JSON embedded in prose), validation against the existing Pydantic
  schemas, and exactly one bounded correction attempt. Invalid structured
  output is never accepted; persistent failures count as model-output
  errors and trigger the normal fallback.
- Tool-calling behavior is not guaranteed to be identical across providers;
  the graph validates tool arguments independently and keeps `create_task`
  behind the human-approval workflow regardless of provider.

**Free-tier usage limitations and data protection:**

- Free models are rate-limited (per-minute and per-day caps) and may be
  temporarily unavailable or slower than paid endpoints; 429 rate-limit and
  transient errors automatically use the Anthropic fallback when
  configured.
- Requests are subject to a context-size limit; the app's existing context
  budgeting (`MAX_CONTEXT_CHARS`, history trimming) applies unchanged.
- The OpenRouter listing for free models includes a notice that data sent
  to the free endpoint may be collected and used. **Do not send confidential
  documents, personal information, private notes, or sensitive long-term
  memory to this model.** Use the synthetic `samples/` corpus or other
  non-sensitive material while this provider is active. Local indexing and
  storage are unchanged; only retrieved context is sent to the model.


### 9.2 Cloudflare Workers AI (GLM-4.7-Flash) setup — the primary provider

The application uses **GLM-4.7-Flash** (by Zhipu AI) through **Cloudflare
Workers AI** as its **primary** provider (follow-up task 4, section 5).
OpenRouter (NVIDIA Nemotron 3 Ultra) is the first fallback; the Anthropic
fallback stays the final resort. Default chain:

```text
Cloudflare Workers AI (GLM-4.7-Flash, primary)
  -> OpenRouter (NVIDIA Nemotron 3 Ultra)   [on rate limit / quota / timeout]
  -> Anthropic (Claude Haiku 5.5)           [legacy fallback, final resort]
```

**Model identifier (exact):**

```text
@cf/zai-org/glm-4.7-flash
```

served through Cloudflare's OpenAI-compatible endpoint

```text
https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1/chat/completions
```

using the existing `langchain-openai` `ChatOpenAI` integration (no new
dependencies). The full base URL is constructed from
`CLOUDFLARE_BASE_URL` + `CLOUDFLARE_ACCOUNT_ID` + `/ai/v1`.

**How to obtain your Cloudflare credentials:**

1. Create a (free) Cloudflare account at <https://dash.cloudflare.com/>.
2. In the Cloudflare Dashboard, go to **Workers AI** and open the
   **Use REST API** page.
3. Click **Create a Workers AI API Token**. If you create a *custom* token
   instead, grant it the **Workers AI** permissions required by the
   official documentation (Workers AI Read at minimum for inference).
4. Copy the **API token** into `.env` as `CLOUDFLARE_API_TOKEN`.
5. Copy your **Account ID** (shown on the same page, and on any domain's
   Overview page) into `.env` as `CLOUDFLARE_ACCOUNT_ID`.

Never paste credentials into source code — they belong only in your local
`.env` file (which is git-ignored). The application never prints or traces
them.

**Required environment variables** (already in `.env.example`):

```env
CLOUDFLARE_ENABLED=true
CLOUDFLARE_ACCOUNT_ID=<your account id>
CLOUDFLARE_API_TOKEN=<your workers-ai api token>
CLOUDFLARE_MODEL=@cf/zai-org/glm-4.7-flash
CLOUDFLARE_BASE_URL=https://api.cloudflare.com/client/v4/accounts
CLOUDFLARE_STRUCTURED_MODE=prompt

LLM_PRIMARY_PROVIDER=openrouter      # unchanged - OpenRouter stays primary
LLM_FALLBACK_PROVIDERS=cloudflare    # ordered fallback chain
```

**Failover behaviour.** Cloudflare is attempted when the primary fails
with an HTTP 429 rate limit, quota exhaustion, a request timeout, a
transient/retryable server error, or a connection failure. The identical
message list (prompt, conversation context, retrieved RAG passages) and
the identical tool definitions are handed to every attempt, so switching
providers never loses context. Failures that indicate configuration
problems (invalid credentials, invalid model identifiers, malformed
requests) do **not** trigger retries — they raise a clear configuration
error with a Cloudflare-specific hint. Each provider is attempted exactly
once per call; the chain length is the retry bound, so nothing can loop
forever. The order is configurable through `LLM_FALLBACK_PROVIDERS`
(comma-separated); the legacy Anthropic fallback is always appended at
the end when its key is configured.

**Compatibility notes.** GLM-4.7-Flash supports tool calling. For
structured output, the OpenAI-compatible endpoint's `response_format`
JSON-Schema support is not documented for this model, so the application
defaults to `CLOUDFLARE_STRUCTURED_MODE=prompt`: explicit JSON
instructions, robust client-side parsing, and validation against the
existing Pydantic schemas (exactly like the OpenRouter free endpoint).
Invalid structured output is never accepted.

**Verifying the integration with your credentials** — run the probe
script (no credentials needed for the reachability level):

```bash
python3 scripts/probe_cloudflare.py
# with credentials in .env:
CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... python3 scripts/probe_cloudflare.py
```

It checks, against the real endpoint: reachability (unauthenticated
requests must return a structured 4xx), plain chat completion, tool
calling, streaming, and prompt-mode structured output with Pydantic
validation.

### 9.3 Model-driven function calling (the centralized tool registry)

Follow-up task 4 replaces "every request goes through a predetermined
branch" with proper **model-driven function calling**, while LangGraph
stays the workflow controller.

**The centralized tool registry** (`tools/registry.py`) is the single
source of truth.  Every tool registers with a name, a clear description,
a validated Pydantic input schema, and a permission boundary:

| Tool | Purpose | Permission |
|---|---|---|
| `search_documents` | Search the shared document library | Read-only |
| `get_document_content` | Read a specific document's passages | Read-only |
| `list_tasks` | List saved tasks | Read-only |
| `get_current_datetime` | Application clock in `APP_TIMEZONE` | Read-only |
| `web_search` | Search the public web (when configured) | Read-only |
| `read_web_page` | Restricted page extraction (search-result URLs only) | Read-only |
| `verify_claims` | Deterministic claim/evidence comparison | Read-only |
| `create_task` | Create a task | **Restricted — human approval required; never bound to a model** |

**How it works.**  The `general_response` graph node runs the agent loop:

```text
Guardrails -> Context preparation -> LLM with the read-only tools bound
    -> tool call?  -> validate arguments (Pydantic schema, independent of
       the model) -> execute through the application-controlled function
       -> ToolMessage back to the LLM -> more tool calls? -> final answer
```

- Tools are bound through the existing LangChain model gateway
  (`model.bind_tools(...)`) — both Cloudflare GLM-4.7-Flash and the
  OpenRouter Nemotron fallback support OpenAI-compatible tool calling, and
  the identical tool definitions go to every provider attempt.
- The MODEL decides when a tool is needed (including the current
  date/time: `get_current_datetime` is a tool the model calls, not a
  hardcoded branch — see section 12.3).  The application validates every
  argument, executes only registered read-only tools, and refuses unknown
  or restricted tool names.
- Bounded execution: at most `LLM_MAX_TOOL_ITERATIONS` (default 6) model
  turns and `LLM_MAX_TOTAL_TOOL_CALLS` (default 12) tool executions per
  request.  Repeated identical calls are suppressed; when the budget is
  exhausted the model is asked to answer with what it has (never a dead
  loop).  Every assistant tool-call message precedes its tool results, so
  the conversation stays valid for strict OpenAI-compatible providers.
- `LLM_TOOL_CALLING_ENABLED=false` binds no tools (direct answers only).
- Stop (cancellation) is checked before every model call and every tool
  execution — no tool runs after the user pressed Stop.
- The document RAG branch (deterministic retrieval-first answering with
  citation validation) is preserved as its own specialized route; the
  agent loop offers `search_documents`/`get_document_content` as tools for
  everything else.

**Current date/time (follow-up task 4, section 7).**  The old
deterministic `datetime_response` node was removed.  "What day is today?"
is routed to the agent, the model decides to call `get_current_datetime`,
and the tool returns the real application clock (date, time, weekday,
timezone, ISO 8601 stamp).  A deterministic safety net still guarantees
the original bug stays fixed: a date/time-shaped question is never
answered from your documents, and if a model ever answers without calling
the tool, one corrective retry runs — and the real application-clock value
is appended, so no fabricated timestamp ever survives.

### 9.4 Web search (optional AI tool)

`web_search` is an optional tool the MODEL can invoke when a request needs
current or independent public information (verifying a disputed fact,
checking current documentation, finding a recent release).  It is never
run automatically for greetings or ordinary explanations.

**Architecture** (`core/web_search.py`): a provider-independent
`WebSearchService` with a `SearchProvider` interface — Tavily is the
initial integration, and additional providers (Brave Search, ...) can be
added behind the same interface.

**Setup:**

1. Create a free account at <https://app.tavily.com>.
2. Copy an API key from the home page (API keys section).
3. In `.env`: `WEB_SEARCH_ENABLED=true`, `WEB_SEARCH_PROVIDER=tavily`,
   `TAVILY_API_KEY=<your key>`.

Without a key the tool stays gracefully unavailable (it is not even
offered to the model) and ordinary chat, documents, and tasks are
unaffected.  `WEB_SEARCH_MAX_RESULTS` (default 5) and `WEB_SEARCH_TIMEOUT`
(default 15 s) tune each search.

**Result handling:**

- Every result carries title, URL, source domain, excerpt, publication
  date (where available) and a retrieval timestamp — these are the only
  URLs the model may ever cite; inventing a URL never passes validation
  (see 9.5).
- Search output is wrapped in the application's UNTRUSTED-content
  delimiters: external web content is evidence, never instructions.
- `read_web_page` is a restricted page-extraction tool: it accepts ONLY
  URLs returned by `web_search` in the same request, requires http/https
  on standard ports, resolves and blocks localhost/private/loopback/
  link-local addresses (SSRF protection, including post-redirect checks),
  and caps time and size.  No arbitrary URL fetching, ever.

### 9.5 Document cross-referencing and fact-checking

Uploaded documents are evidence of **what the document says**, not
evidence that it is true.  The assistant can now verify that difference.

**Activation.**  Explicit requests such as "Fact-check my uploaded
research notes", "Is the information about chemical elements correct?",
"Check whether my document contains incorrect information", or "Compare
the claims in my PDF against reliable information" route to the dedicated
fact-check workflow (intent `fact_check` with a deterministic safety
net).  Ordinary document questions never trigger it — no expensive
full-document pass on every question.  When answering a normal document
question, the assistant may still note a clear contradiction it happens
to see.

**Workflow** (two LangGraph nodes, `fact_check_claims` ->
`fact_check_verification`):

```text
Retrieve relevant passages -> Extract verifiable claims (verbatim, with
their document/page/section source) -> Compare each claim against the
model's general knowledge -> claims flagged as needing independent
verification -> bounded web searches (only when configured) ->
evidence comparison -> application-side validation -> structured report
```

**Verification categories** (never a forced true/false): `correct`,
`incorrect`, `partially_correct`, `misleading`, `outdated`,
`contradictory`, `unverified`, `opinion`, `ambiguous`.  Opinions ("The
best programming language is Python") are identified as opinions; if
independent evidence is insufficient the claim is marked **unverified**
instead of being confidently declared false.

**Evidence layers stay distinct.**  Every finding separates (A) what the
document says, (B) what the model knows, (C) what independent retrieved
sources support, and (D) the final assessment.  The report UI labels each
finding's verification method (Model knowledge vs. Model knowledge +
independent sources vs. Independent sources) so you can tell which
conclusions rest on the model alone.

**Citation safety.**  External sources are validated application-side
after the final structured call: any cited URL, title, or date that was
not actually retrieved is stripped and noted as a limitation — the model
cannot smuggle an invented citation past the workflow.

**Report.**  Each finding shows the original claim and its source, the
status, the corrected information, the explanation, the document
evidence, the independent sources (linked), confidence, and limitations —
each finding expandable, read-only, and your original documents are never
modified.

## 10. Application startup instructions

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first use, upload documents in
**Documents**, then chat. To seed demo tasks (optional):

```bash
python scripts/seed_demo.py
```

To regenerate the sample corpus: `python scripts/make_sample_docs.py`.

**Sidebar toggle**: the ☰ control sits at the top-left of the sidebar,
next to the brand (tooltip "Close sidebar"). When the sidebar is
collapsed, a reopen control (tooltip "Open sidebar") floats at the left
edge of the application — the same top-left corner in both states, exactly
like the ChatGPT web UI. Its state lives in `st.session_state`
(`pkta_sidebar_open`), survives ordinary reruns, and toggling never resets
the active conversation or interrupts pending approvals — the workflow
state lives in the SQLite checkpointer, not in UI state. The application
state is authoritative: it even re-opens the sidebar if the browser's
persisted Streamlit state disagrees.

**Asking for the current date/time**: "What day is today?" and similar
questions are answered by a deterministic utility (`core/datetime_util.py`)
reading the application clock in the configured timezone
(`APP_TIMEZONE`, default `Asia/Dhaka`, validated at startup). No model
call and no document retrieval is involved, so the answer can never be
blocked by the document-abstention mechanism and never comes from stale
training data.

## 11. How to upload documents

Upload surfaces (all feed the same shared library):

1. **Chat page (empty chat)** — a large drop zone in the middle of the
   screen; drag files onto it or click to browse.
2. **Chat page (ongoing conversation)** — the 📎 **Attach** button next to
   the message composer opens a drop zone.
3. **Documents page** — the full-page uploader with the library list.

Each upload shows live staged progress (never fake percentages):

```
Uploading notes.pdf...
        |
Validating file ...
Checking for duplicates ...
Extracting text ...
Generating chunks ...
Generating embeddings ...
Indexing in Qdrant ...
Document ready — 12 chunks · 25 pages · 0.11s
Stage timings — parse 0.02s · embed 0.08s · index 0.01s · total 0.11s
```

Behaviour details:

- Supported: PDF, DOCX, Markdown, TXT (max 20 MB, 10 indexed documents).
- **Duplicate uploads are skipped instantly** (content SHA-256 check) —
  re-dropping the same file costs nothing and never creates a second
  record.
- Uploading a **changed file under the same filename replaces** the old
  version (the previous chunks are removed first).
- A file that fails to parse (e.g. a scanned PDF) is reported with a
  descriptive error **without touching** any previously indexed version.
- **Remove** (with confirmation) deletes the vector chunks and the library
  entry for every chat.

The synthetic corpus in `samples/` (study plan, ML notes, project spec,
research summary, learning goals) is ready to upload for a first demo.

### Performance notes

Measured with `scripts/verify_pipeline.py` (real model, real Qdrant,
  real workflow; see section 18.1):

| Document | Chunks | Total time |
|---|---|---|
| Small TXT | 1 | ~0.03 s |
| Small PDF (1 page) | 1 | ~0.23 s (first PDF parse imports pypdf) |
| Small DOCX | 1 | ~0.02 s |
| Small Markdown | 1 | ~0.01 s |
| 25-page PDF | 25 | ~0.09 s |
| Duplicate re-upload | — | ~0.00 s (skipped) |

The embedding model loads **once** per process (~8 s warm-up from local
  cache) in a background thread at startup, so the first upload no longer
  pays that cost. If the model was never downloaded before, the first-ever
  run downloads ~90 MB from Hugging Face.

### 11.1 Uploading documents through the chat (plus button and drag-and-drop)

The chat composer supports two upload paths, both feeding the SAME shared
document library and ingestion pipeline as the Documents page:

- **Plus button** — the "＋" control left of the message input is a
  single-click file browser (no popover, no extra click). Selecting files
  starts ingestion immediately: a staged progress panel shows validation,
  duplicate detection, parsing, chunking, embedding, and Qdrant indexing
  with real per-stage timings, and confirms when the document becomes
  searchable.
- **Drag and drop** — drop supported documents (PDF, DOCX, TXT, Markdown)
  anywhere on the chat/message area. A clear drag-over overlay confirms
  the target; on drop the same validation and ingestion pipeline runs.

Each uploaded file appears as an **attachment chip above the message
input**, showing the file-type badge, filename, and indexing status.
Clicking a chip removes it from the preview only — the document stays in
the globally shared library (delete it on the Documents page if you want
it gone everywhere). Uploaded documents are immediately searchable in
**every** chat session through RAG; they are never tied to a single
thread. Duplicate uploads (same name + size + content) are detected and
skipped, and the library keeps all existing management features
(deletion, re-indexing, persistence).

## 12. How to ask questions using RAG

In the **Chat** page, ask in natural language, e.g.:

> Which vector database did I choose for my project?

**Retrieval is executed by the application, not by the model.** The
`document_retrieval` node runs a deterministic retrieval-first pipeline:

1. **Empty-library check** — if no documents are indexed at all, you get an
   explicit "no documents are currently indexed" answer with upload
   guidance (an application-state message, distinct from *insufficient
   evidence* and from errors).
2. **Query resolution** — self-contained questions are used verbatim;
   contextual follow-ups ("What was the reason for choosing it?") are
   resolved into standalone questions using the conversation context (one
   bounded model call, with a deterministic fallback that merges the
   previous user question when the provider is unavailable).
3. **Direct retrieval** — the application searches the shared Qdrant
   collection with the resolved query. This always runs, before any
   planner LLM call.
4. **Optional planner supplement** — only when direct retrieval was weak
   (fewer than 2 chunks), a bounded read-only tool loop may gather more
   passages. A planner failure (rate limit, outage) is **non-fatal**: the
   direct results still reach the answer generation.
5. **Grounded answer + citation validation** — the model answers ONLY from
   the retrieved passages (wrapped in untrusted-content delimiters); every
   citation is validated against the actually-retrieved chunks.

**You never need to invoke a search tool yourself.** Three deterministic
safety nets (application code, not prompts) keep document questions
connected to the RAG branch even when the free-tier classifier misbehaves:

1. If the classifier labels a document-shaped question as
   `general_conversation` (or `unsupported_request`) while the library has
   indexed documents, the intent is overridden to `document_question`.
   This covers both explicit references ("my uploaded PDF", "my project
   proposal") and question shapes ("Where is X discussed?", "Compare the
   two …", "summarize", "main points", …). Task operations are exempt.
2. A contextual follow-up inside a thread that is already answering
   document questions stays on the document branch (so the follow-up above
   retrieves again instead of being treated as small talk).
3. If the classification call itself fails (rate limit, outage), a
   deterministic heuristic classifier routes the request instead of failing.

**Distinct failure semantics** (never confuse these):

| Situation | Response |
|---|---|
| Library empty | "no documents are currently indexed …" + upload guidance |
| Nothing relevant found | Abstention: "I could not find sufficient information about this in the documents currently indexed." |
| Qdrant / embedding / storage failure | An explicit **error** response ("Document search is currently unavailable …") and a `[rag-diag]` log line |
| Model outage during answering | Error response via the normal gateway fallback policy |

**Shared library semantics**: upload a document in Chat 1 and ask about it
in Chat 2, Chat 3, or any future chat — retrieval searches one global
collection, not the current conversation. Conversation histories stay
isolated; only document access is shared. Removing a document removes it
for every chat.

**Diagnostics**: run `streamlit run app.py` and watch the terminal — every
upload logs `[rag-diag] ingestion finished: … collection points now: N`,
and every document question logs the routing decision, the retrieval
execution, the context size that entered the LLM request, the citation-
validation outcome, and the final answer outcome. No document text or
credentials are logged. Set `PKTA_RAG_LOG=DEBUG` for more,
`PKTA_RAG_LOG=WARNING` to quiet.

**Development-only request capture**: "context included" diagnostics prove
the messages were *built*; to prove what was actually *sent*, set
`PKTA_DEV_CAPTURE_DIR=/some/dir` before starting the app — every chat and
structured call then appends its final message list (full contents) to
`model_messages.jsonl` immediately before the provider call. Because this
captures document content, it is disabled unless the variable is set and
must never be enabled in production. The automated test suite exercises
the same hook (`tests/test_followup2_acceptance.py`, Test B).

## 13. How citations work

Every answer displays its **Sources** as expandable entries:

```
Source 1: StudyPlan.pdf — Page 3
```

Expanding it shows the supporting passage, the chunk id, and the retrieval
score. Behind the scenes:

1. Each retrieved chunk carries authoritative metadata (document id,
   filename, page number for PDFs, heading for DOCX/Markdown, chunk id).
2. The model must cite chunk ids from the passages it was given.
3. `guardrails.validate_citations` **drops** any citation whose chunk id was
   not actually retrieved and overwrites the metadata with the chunk's real
   values (page numbers are never invented).
4. If no valid citation remains — or the model flags insufficient evidence —
   the assistant replies: *"I could not find sufficient information about
   this in the documents currently indexed."* (When the library is empty
   altogether, a distinct message tells you that nothing has been uploaded
   yet, so missing evidence is never confused with missing documents.)

DOCX/Markdown/TXT files have no real page numbers, so they are cited by
filename and heading.

## 14. How task creation and human approval work

1. Say something like *"Add a task to prepare for the ML exam on December
   18, high priority."*
2. The `task_proposal` node generates a **TaskProposal** (title,
   description, due date, priority, confirmation message) validated by
   Pydantic. Missing information is never fabricated.
3. The workflow calls `interrupt()` at `human_approval` and **pauses**. The
   chat and the Approval Center show the proposal with **Approve** /
   **Reject** buttons and the warning that nothing has been saved yet.
4. **Approve** resumes the graph with `Command(resume="approved")` → the
   `task_execution` node invokes the restricted `create_task` tool → the
   task is written.
5. **Reject** (or any ambiguous outcome) discards the proposal; nothing is
   written.

Idempotency: every proposal carries a unique `operation_id`; the tasks table
enforces `UNIQUE(operation_id)`. Double-clicking Approve, refreshing the
page, or resuming a completed thread never creates a duplicate task. The
model itself can never invoke the write tool — it is not bound to any model.

## 15. Conversation memory and multiple chats

Each conversation has a stable `thread_id`. All messages and workflow state
are checkpointed to `checkpoints.db` (LangGraph SQLite checkpointer), so:

- follow-ups like *"what is its due date?"* resolve from context,
- switching between conversations keeps them isolated,
- refreshing the browser or restarting the app loses nothing,
- a pending approval survives restarts and can be resumed afterwards.

**Chat management** (sidebar):

- **＋ New chat** starts a completely fresh conversation: a new `thread_id`,
  an empty transcript, no shared state except the document library and
  long-term preferences. Existing chats, their checkpoints, the vector
  index, and saved preferences are never cleared.
- **Chat history** lists every conversation, most recently active first,
  with the active chat highlighted. The title comes from the first user
  message (trimmed to 48 characters on a word boundary — no extra model
  call is used for titles).
- **× / Delete chat** removes a conversation after an explicit confirmation.
  Deletion removes only the conversation entry: the shared document library,
  tasks, preferences, other chats, and the thread's checkpoints (needed to
  resume a pending approval through the Approval Center) are preserved.

Conversation memory is deliberately separate from long-term memory: the
former is per-thread workflow state; the latter is a cross-thread
preferences store. Document access, in contrast, is **shared** across all
threads (section 12).

## 16. Long-term memory explanation

On the **Settings** page you explicitly save preferences (key/value pairs,
e.g. `default_task_priority: high`). They are stored in the `user_memory`
SQLite table and:

- injected into every new conversation in every thread,
- available to the model for defaults (e.g. task priority),
- viewable and deletable at any time,
- **never** written automatically from chat, and
- **never** store secrets: credential-like keys and values are rejected.

Demonstration: save a preference in one conversation, start a brand-new
conversation, and the preference is already in effect.

## 17. LangSmith tracing setup

1. Create a free account at <https://smith.langchain.com/> and copy an API
   key.
2. In `.env`, set `LANGSMITH_API_KEY=...`, `LANGSMITH_TRACING=true`, and
   `LANGSMITH_PROJECT=personal-knowledge-task-assistant`.
3. Restart `streamlit run app.py`.

LangChain then traces automatically: model calls (with run names such as
`classify_intent`, `grounded_answer`, `agent_response`,
`fact_check_extraction` / `fact_check_assessment` / `fact_check_report`),
the LangGraph node executions, retrieval and tool calls, and errors.
Fallback events appear in the metadata of the fallback run
(`attempt: fallback_1`, `fallback_reason: <provider>_<category>`) and in
the run's error fields.

Follow-up task 4 extends the same metadata to function calling: every
`agent_response` run records the selected provider and model, the tools
offered, the tool iterations used, and the tool calls completed so far;
each tool execution records its name, duration, success/failure, and (for
web search) the serving provider in the tool trace.  Fact-check runs
record claim counts, web-query counts, and the evidence-validation
outcome.  No credentials and no API keys ever appear in traces.

Privacy: document content appears in traced inputs by default. Set
`LANGSMITH_HIDE_INPUTS=true` to mask all run inputs, or disable tracing by
setting `LANGSMITH_TRACING=false` — the application is fully functional
without LangSmith.

Privacy: document content appears in traced inputs by default. Set
`LANGSMITH_HIDE_INPUTS=true` to mask all run inputs, or disable tracing by
setting `LANGSMITH_TRACING=false` — the application is fully functional
without LangSmith.

## 17.5 Stopping a response (request cancellation)

While the assistant is generating, the Send button is **replaced by a
Stop button** (square stop icon) in the same position, and the input
shows a "Generating response..." placeholder. Pressing Stop:

1. signals cancellation to the active generation,
2. stops further model calls, tool executions, and task writes at the
   next safe point (nodes check the cancellation flag at entry, inside
   tool loops, and immediately before the irreversible `create_task`
   write),
3. marks the assistant message as an explicit stopped response,
4. restores the Send button and re-enables the input.

Generation runs in a background worker thread so the interface stays
interactive while the model works. One honest limitation of the installed
synchronous clients (`langchain-openai` / `langchain-anthropic` on the
synchronous invoke path): a single in-flight HTTP request cannot be
aborted mid-flight — the current call finishes (bounded by
`MODEL_TIMEOUT_SECONDS`, at most 90 s), its output is discarded, and
every subsequent step is skipped. The live progress label under the
thinking dots ("Searching uploaded documents...", "Retrieving relevant
passages...", "Preparing response...", "Stopping...") reflects actual
application events, not fabricated reasoning.

A message submitted while a stopped run is still winding down is queued
and sent automatically as soon as the worker exits — nothing is lost, and
no two runs ever share a conversation thread. Cancelled runs never
corrupt the checkpointer: the graph completes its normal path with a
"stopped" final response, so the transcript stays coherent (your message
plus the stopped marker) and the thread remains fully usable.

## 18. How to run automated tests

```bash
python -m pytest
```

Over 250 tests run in a few seconds, fully offline — fake models,
deterministic fake embeddings, temporary databases and vector stores. The
suite covers: structured-output validation (including the prompt-based
structured-output path used for OpenRouter: fenced JSON, malformed replies,
bounded correction, fallback), the model gateway (success, each failure
category, both-down, malformed output, provider selection and switching),
all three tools, the RAG pipeline (parsing, chunking, metadata, vector
round-trip, citation validation, abstention), all workflow branches,
conversation memory and thread isolation, durable approval (pause, approve,
reject, double-approve, restart-resume, idempotency), long-term memory
(save, cross-thread use, delete, secret rejection), **ingestion
deduplication and stage timings** (duplicate skip, filename replacement,
corrupt-upload safety, legacy-database migration, batched upserts), the
**shared document library** (cross-chat retrieval, global deletion,
history isolation), **chat-session management** (titles, deletion
semantics, approval survival), the **deterministic routing safety nets**,
and — in `tests/test_document_access_fix.py` — the follow-up regression
suite: ingestion with real point counts, direct retrieval, every task
example question reaching the document branch under a misbehaving
classifier, retrieved text inside the LLM request, planner-outage
resilience, infrastructure-failure-as-error (never silently "no
evidence"), contextual follow-up query resolution (model rewrite and
deterministic fallback), citation origin, insufficient evidence, restart
persistence, safe diagnostics logging, and sidebar-restoration canaries.

Follow-up task 4 adds dedicated suites (still fully offline):

- `tests/test_tool_registry.py` — the centralized registry: required tool
  set, permission boundaries, `create_task` never bindable, unavailable
  web search excluded, schema validation of every bindable tool.
- `tests/test_agent_tools.py` — the model-driven loop: tool selection and
  continuation, multiple tool calls per message, strict conversation
  validity (assistant tool-call message precedes its tool results),
  invalid-argument feedback, unknown-tool refusal, duplicate-call
  suppression, iteration-budget exhaustion, tool-calling disabled, and
  document tools used by the model.
- `tests/test_datetime_tool.py` — "What day is today?" / "What time is it
  in my timezone?" answered through model-invoked `get_current_datetime`
  with the real application clock; fabricated-timestamp rejection;
  removal of the deterministic datetime node; documents never consulted.
- `tests/test_web_search.py` — the provider-independent service, result
  metadata, untrusted wrapping, provider errors, graceful
  unavailability, the restricted page-extraction allowlist, and SSRF
  protections (localhost/private/link-local hosts).
- `tests/test_fact_check.py` — the exact synthetic chemical-symbol
  scenario from the task (gold=Ag -> incorrect with the Au/Ag
  explanation; silver=Ag -> correct; water=H2O -> correct), passages
  reaching the extraction call as untrusted content, web evidence for
  flagged claims, invented-URL stripping, misrouting safety net, ordinary
  questions never fact-checking, and opinion categorization.
- `tests/test_tool_safety.py` — `create_task` never executed through the
  agent loop, task creation still requiring human approval (ambiguous
  resume never approves), document and web prompt-injection resistance,
  independent argument validation, restricted page fetching.
- `tests/test_provider_tool_calling.py` — Cloudflare-primary /
  OpenRouter-first-fallback chain built from settings, and tool-call
  context preserved across a provider switch.
- `tests/test_followup4_ui.py` — the UI contracts: plus-icon absolute
  centering, sidebar top spacing, toggle/brand alignment, centered
  suggestion buttons, the shared composer draft key, insert-without-
  submit behaviour, and the fact-check report structure.

Your real `data/` directory is never touched.

To verify the live OpenRouter endpoint once your key is configured, run
`python scripts/verify_openrouter.py` (provider initialization, basic
conversation, structured output, tool calling — using synthetic,
non-sensitive content only).

### 18.1 End-to-end pipeline verification (real embeddings)

```bash
python scripts/verify_pipeline.py
```

Runs the real ingestion pipeline (real sentence-transformers model, real
Qdrant local storage, real LangGraph workflow; a scripted stand-in for the
language model) in a temporary data directory and verifies, with printed
stage timings: fast indexing of small TXT/PDF/DOCX/Markdown files and a
multi-page document, one-time embedding-model loading, instant duplicate
skip, **cross-chat retrieval reaching the answer prompt in every thread**,
and citation validity, abstention on unsupported questions, and
conversation isolation with shared document access. It downloads the
embedding model from Hugging Face on the very first run (~90 MB).

## 19. Known limitations

- **In-flight model requests cannot be aborted mid-HTTP-call** — the
  installed synchronous LangChain clients expose no cancellation handle;
  Stop takes effect at the next safe point (see section 17.5). The
  current call is bounded by `MODEL_TIMEOUT_SECONDS`.
- **Cloudflare live-model tests require credentials** — the automated
  suite verifies configuration, URL construction, error classification,
  failover order, and context preservation offline; the real GLM-4.7-Flash
  round-trips are verified by `scripts/probe_cloudflare.py` when
  `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are set (the
  reachability level needs no credentials).
- **Web search is optional and external** — without a `TAVILY_API_KEY` the
  `web_search` tool stays unavailable (ordinary chat and documents are
  unaffected); with one, searches go to the Tavily API and the snippets
  are evidence only.  Fact-check findings that need independent evidence
  are marked unverified (with a note) when web search is off.
- **Fact-checking is bounded by design** — at most
  `FACT_CHECK_MAX_CLAIMS` claims are extracted per request and at most
  `FACT_CHECK_MAX_WEB_QUERIES` external searches run; longer documents
  are verified through their most relevant passages, not every sentence.
  The workflow activates only on explicit verification requests.
- **Model-driven tool calling depends on the provider's function-calling
  support** — both the primary (Cloudflare GLM-4.7-Flash) and the first
  fallback (OpenRouter Nemotron) support OpenAI-compatible tool calling
  and receive identical tool definitions; if a provider in your chain
  lacks tool support, its answers arrive without tool use (the gateway
  never silently rewrites requests).
- **Streaming into the chat UI** — the Stop/progress design executes the
  workflow to completion before the transcript renders the response;
  token-by-token streaming into the message bubble is not implemented
  (the composer-level progress indicator shows the real phase instead).

- **English documents** are assumed for retrieval quality; other languages
  are not tested.
- **Scanned/image-only PDFs** cannot be indexed (no OCR).
- **Updating or deleting existing tasks** is not supported in this version —
  only listing, filtering, and creating after approval. Requests to modify
  tasks are politely refused.
- **Pattern-based guardrails** can occasionally refuse a legitimate question
  that *mentions* a banned action (e.g. "what do my notes say about shell
  scripts?"); phrasing the question differently resolves it.
- **Deterministic routing override** fires for messages that refer to your
  documents (broad noun list) or match document-question shapes ("where is X
  discussed", "compare the two", "summarize", "main points", …) while at
  least one document is indexed, and for contextual follow-ups in threads
  already answering document questions. Questions with *no* document
  reference at all (e.g. "Where is authentication discussed?" with nothing
  indexed about it) still rely primarily on the LLM classifier; if that
  misclassifies such a question, the honest result is an abstention, not a
  fabricated answer.
- **Small-corpus retrieval**: with only a few documents, paraphrased
  questions may miss the passage; the assistant then abstains (by design).
- **Sidebar chat history** renders one button per conversation; with a very
  large number of chats (hundreds) the sidebar becomes long — scrollable,
  but not paginated.
- **Deleting a chat keeps its LangGraph checkpoints** (on purpose, so a
  pending approval stays resumable); the checkpoint rows are small but not
  garbage-collected with the conversation.
- **Single user, local only** — no auth, no sharing, no cloud deployment.
- The fallback path is one attempt per provider; sustained outages surface a
  clear error instead of retry loops.
- **Free-tier OpenRouter model (temporary)** — rate limits, temporary
  unavailability, and the data-collection notice described in section 9.1
  apply while `LLM_PRIMARY_PROVIDER=openrouter`. Structured output relies
  on prompt instructions plus client-side validation rather than native
  schema enforcement (see section 9.1).

## 20. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Red banner "No language model API key is configured" | Copy `.env.example` to `.env` and set at least one key. |
| Red banner "LLM_PRIMARY_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set" | Add your OpenRouter key (https://openrouter.ai/keys) or switch back with `LLM_PRIMARY_PROVIDER=openai`. |
| Warning "the original primary model ... is configured but not active" | Informational: `LLM_PRIMARY_PROVIDER=openrouter` is active; set `LLM_PRIMARY_PROVIDER=openai` to restore GPT-6 Luna. |
| 429 / rate-limit errors from the free Nemotron model | Free-tier rate cap; the Anthropic fallback handles the request automatically when configured, or wait and retry. |
| Warning "fallback model is disabled" | Set `ANTHROPIC_API_KEY` to enable the fallback. |
| First upload is slow | The embedding model downloads once (~90 MB). It is warmed up in a background thread at startup, so even that first upload only waits for the download. Subsequent runs are fast. |
| A message that clearly refers to my documents got a generic answer | First check the terminal: the `[rag-diag]` lines show whether retrieval ran and how many chunks it returned. If retrieval found nothing relevant, the assistant honestly abstains — rephrase the question or check that the document is indexed on the Documents page. If the answer disclaims document access entirely, the question had no document reference the safety nets recognize; mention the document ("my proposal", "the PDF", "my notes", "where is X discussed …"). |
| I want to verify indexing really works | Watch the `[rag-diag]` log lines while uploading and asking (collection, points, retrieved chunk count, context size in the LLM request), or check the shared-collection point count on the Settings page. |
| The sidebar disappeared and I cannot restore it | Use the ☰ menu button in the slim application header at the top of the main area (tooltip: "Open sidebar"). It is rendered outside the sidebar, so it is always reachable. |
| "testdoc.txt is already indexed — skipped" | Expected: the identical file (same content hash) is already in the shared library; nothing was duplicated. |
| Chat disappeared after deletion but its pending approval is still in the Approval Center | By design: deleting a conversation keeps checkpoints so pending approvals stay resumable. Resolve the approval there, or ignore it. |
| "The existing vector index was built with a different embedding dimension" | You changed `EMBEDDING_MODEL`. Delete the `data/qdrant` folder (and re-upload), or point `QDRANT_PATH` at a new directory. |
| "Scanned/image-only PDFs are not supported" | The PDF has no text layer; use a text-based PDF. |
| Model error mentioning `gpt-6-luna` or `claude-haiku-5-5` not found | Your account may not have access; set `PRIMARY_MODEL` / `FALLBACK_MODEL` in `.env` to a model you can call. The gateway reports this instead of silently switching. |
| Pending approval disappeared after an error | The approval index marks failed resumes; ask the assistant to propose the task again. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502`. |
| Tests fail with import errors | Run `pip install -r requirements.txt` and run pytest from the project root. |

---

## Security notes

- `.env` is git-ignored; never commit keys.
- Retrieved document content is wrapped in UNTRUSTED markers and treated as
  data, never instructions.
- While the free OpenRouter model is active, avoid indexing or asking
  questions about confidential documents or personal data (free-endpoint
  data-collection notice; see section 9.1).
- The only database write reachable from a conversation is task creation,
  and only after your explicit approval, with a unique operation id.
- No SQL, shell, filesystem, or network tools are exposed to the model.
