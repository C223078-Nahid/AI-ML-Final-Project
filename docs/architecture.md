# Architecture

This document explains the runtime architecture of the Personal Knowledge &
Task Assistant: graph nodes, conditional edges, tool boundaries,
persistence, the retrieval process, and the approval workflow.

## 1. Runtime layers

```
┌────────────────────────────────────────────────────────────┐
│ Streamlit UI (app.py, ui/*)                                │
│   chat · documents · tasks · approvals · settings          │
├────────────────────────────────────────────────────────────┤
│ LangGraph StateGraph (graph/*)  — the application controller │
│   nodes + conditional edges + interrupt()                   │
├──────────────┬───────────────┬──────────────────────────────┤
│ core/*       │ tools/*       │ rag/*                        │
│ gateway      │ search (RO)   │ parsers → chunking →         │
│ prompts      │ list (RO)     │ embeddings → vector store    │
│ guardrails   │ create (RO*)   │ → retrieval → ingestion      │
│ schemas      │                │                              │
├──────────────┴───────────────┴──────────────────────────────┤
│ database/* (SQLite: app.db)  ·  graph/persistence.py        │
│ tasks · user_memory · documents · conversations ·           │
│ pending_operations          ·  checkpoints.db (LangGraph)   │
└────────────────────────────────────────────────────────────┘
        * create_task is a write tool, restricted to the approved path
```

Dependency rule: UI → graph → core/tools/rag → database. Nothing in
`database/` imports from the UI or the model layer, and the LLM never sees
SQL.

## 2. Graph nodes

| Node | Responsibility | Model calls | Tools |
|---|---|---|---|
| `input_guardrail` | Validate message length/patterns; load long-term preferences; reset per-turn state | none | none |
| `intent_router` | Classify the request into exactly one `Intent` (structured output) | `IntentResult` | none |
| `document_retrieval` | Retrieval-first: empty-library check, context-aware query resolution, direct retrieval by the application, optional bounded planner supplement (non-fatal on failure) | chat (query rewrite, optional planner) + tool binding | `search_documents` (RO) |
| `grounded_answer` | Evidence-based answer, citation validation | `GroundedAnswer` | none |
| `task_listing` | Deterministic listing with optional status filter | none | `list_tasks` (RO) |
| `task_proposal` | Generate + validate a `TaskProposal`, mint `operation_id` | `TaskProposal` | none |
| `human_approval` | **`interrupt()`** — durable pause; explicit approve/reject only | none | none |
| `task_execution` | Invoke restricted `create_task` (idempotent) | none | `create_task` (WO) |
| `general_response` | Conversation with a bounded read-only tool loop | chat + tools | `search_documents`, `list_tasks` (RO) |
| `final_response` | Normalize the UI payload, append the assistant message, touch the conversation index | none | none |

## 3. Conditional edges

```
input_guardrail → route_after_guardrail
    errors or refusal payload  → final_response
    otherwise                  → intent_router

intent_router → route_after_intent
    document_question     → document_retrieval
    list_tasks            → task_listing
    create_task           → task_proposal
    general_knowledge     → general_response
    general_conversation  → general_response
    current_datetime      → datetime_response (deterministic app clock)
    unsupported_request   → final_response   (refusal text)
    unknown/malformed     → general_response (never RAG by default)

document_retrieval → route_after_retrieval
    errors or ready final payload (empty library)
                          → final_response
    otherwise             → grounded_answer

task_proposal → route_after_proposal
    invalid/failed proposal → final_response (asks for clarification)
    valid proposal          → human_approval

human_approval → route_after_approval
    approval_status == "approved" → task_execution
    anything else (rejected, ambiguous) → final_response
```

Every model-calling node catches `ModelGatewayError`, records the error in
state, and routes to `final_response`, which renders a friendly message —
the graph never crashes on provider failures.

## 4. Tool boundaries (why writes are safe)

- `search_documents(query)` — read-only Qdrant query. Bound to models in the
  two agent loops. Records every returned chunk into a per-run collector,
  which is the ground truth for citation validation.
- `list_tasks(status)` — read-only SQLite query. Bound to models; also
  invoked deterministically by `task_listing`.
- `create_task(title, description, due_date, priority, operation_id)` — the
  **only** write tool. It is never bound to any model and never present in
  any ToolNode: only `task_execution` invokes it, and `task_execution` is
  reachable exclusively from `human_approval` resumed with an explicit
  `"approved"` value. The tool validates its input against Pydantic
  (`CreateTaskInput`) independently of the model, and the repository enforces
  `UNIQUE(operation_id)`.

Argument validation for read-only tools happens through LangChain's
`args_schema`; invalid tool calls are fed back to the model as error strings
inside the loop instead of crashing the workflow.

## 5. State and persistence

`WorkflowState` (TypedDict) fields: `messages` (reducer `add_messages`,
accumulates per thread), `thread_id`, `user_request`, `intent`,
`retrieved_documents`, `task_proposal`, `operation_id`, `approval_status`,
`task_id`, `tool_results`, `preferences`, `final_response`, `errors`.
`input_guardrail` resets all per-turn fields at each new message, so state
never leaks across turns; `messages` intentionally does.

Persistence uses two separate SQLite files:

- **`app.db`** — application data: `tasks`, `user_memory`, `documents`,
  `conversations`, `pending_operations`. The last two are UI-facing indexes
  (thread registry and approval center) that complement, not replace, the
  graph state.
- **`checkpoints.db`** — the LangGraph checkpointer (SqliteSaver). Stores the
  full state of every thread, including the position of a paused run, which
  is what makes `interrupt()`/resume durable across restarts.

All values stored in state are JSON-serializable (or LangChain messages), so
checkpoints round-trip cleanly; the test suite runs with
`LANGGRAPH_STRICT_MSGPACK=true` to enforce this.

## 6. Retrieval process (RAG)

1. **Ingestion** (`rag/ingestion.py`): validate (extension, size) →
   **deduplicate** (SHA-256 content hash; already-indexed content is
   skipped instantly, and the capacity check only applies to genuinely new
   documents) → parse (`pypdf` per page / `python-docx` by heading /
   markdown by heading / txt — before any destructive step, so a corrupt
   upload never removes an indexed version) → chunk (sentence-aware,
   ~700 tokens, ~100 overlap, chars/4 token estimate) → embed locally
   (`all-MiniLM-L6-v2`, 384-dim, normalized, batched, loaded once and warmed
   up in a background thread at startup) → upsert into Qdrant local storage
   in bounded batches with payload metadata (document id, filename,
   page/section, chunk id, upload time). Every stage is timed
   (`IngestionResult.timings`) and surfaced in the UI; re-uploading a
   changed file under the same filename replaces the old version. Documents
   table records status, chunk count, page count, content hash, and errors.
   The collection is a single global `documents` collection shared by every
   chat thread.
2. **Retrieval (retrieval-first)**: the application — not the model —
   executes retrieval. The node first answers the empty-library case with a
   distinct message. It then resolves the search query: contextual
   follow-ups ("What was the reason for choosing it?") become standalone
   questions via one bounded model call, with a deterministic fallback that
   merges the previous user question when the provider is unavailable.
   Direct retrieval with the resolved query ALWAYS runs. Only when direct
   retrieval returns fewer than 2 chunks does an optional bounded planner
   loop (≤ `MAX_TOOL_ITERATIONS` calls, ≤ `MAX_TOTAL_TOOL_CALLS` total)
   supplement with reformulated `search_documents` calls — and a planner
   failure (rate limit, outage) is non-fatal, because the direct results
   already exist. Retrieval infrastructure failures (Qdrant, embeddings)
   surface as workflow errors, never silently as "no evidence".
3. **Grounding**: passages are rendered inside explicit UNTRUSTED markers
   with `SOURCE` headers (source_id, file, page/section). The
   `GroundedAnswer` prompt forbids outside knowledge and invented metadata,
   and includes the context-resolved standalone question as an
   interpretation hint for follow-ups.
4. **Citation validation** (`core/guardrails.py`): citations are kept only
   if their `chunk_id` and `document_id` match a retrieved chunk; kept
   citations have filename/page/section overwritten from the chunk payload;
   quoted passages are checked against the chunk text (non-quotes are
   replaced with a real excerpt); zero valid citations or a model-flagged
   `insufficient_evidence` becomes the canonical abstention message.
5. A retrieval score threshold (`RETRIEVAL_MIN_SCORE`, cosine) filters
   near-noise matches on the semantic path; the **hybrid retrieval**
   (follow-up task 3.4) adds a SQLite FTS5 keyword fallback
   (`rag/lexical_index.py`) that finds exact facts ("What is my name?",
   score 0.34 with all-MiniLM-L6-v2 on a marginal match) even when the
   embedding distance is above the threshold. Results are merged and
   deduplicated by chunk id and normalized text; the bounded planner loop
   now runs only when the direct (hybrid) retrieval is EMPTY, which also
   removed the planner-induced duplicate retrieval counts.
6. **Routing safety nets** (`graph/nodes.py::intent_router`): (a) a message
   that refers to the user's documents (broad noun list: documents, PDFs,
   TXT files, proposals, reports, notes, …) or matches a document-question
   shape ("Where is X discussed?", "Compare the two …", "summarize", "main
   points", personal-fact lookups such as "What is my name?", …) while the
   shared library has indexed documents is re-routed to the RAG branch even
   if the classifier answered `general_conversation`, `general_knowledge`
   or `unsupported_request` (task operations are exempt); (b) a contextual
   follow-up in a thread that already produced grounded answers stays on the
   document branch; (c) **datetime safety net** — a question about the
   current date/day/time is forced to the deterministic `datetime_response`
   node (never RAG, never model training data) regardless of the
   classifier's label, fixing the reported "What day is today?" abstention;
   (d) if the classification call fails entirely, a datetime-first
   deterministic heuristic classifier routes the request instead of erroring
   out. All are pure application code in `core/guardrails.py` and are
   unit-tested. Safe `[rag-diag]` diagnostics (collection, points, chunk
   counts, context size, provider/model, routing decision, citation
   validation, final outcome — never content) are logged by
   `rag/diagnostics.py` at every ingestion, retrieval, and answer.

## 7. Approval workflow (durable pause/resume)

```
task_proposal (validates, mints operation_id)
        │
human_approval:  decision = interrupt({operation_id, proposal, question})
        │                 ── workflow PAUSED, checkpoint saved ──
        │ return control to Streamlit; proposal recorded in
        │ pending_operations; chat input disabled; Approval Center shows it
        │
   ┌────┴─────────────────────────────┐
   │ resume with Command(resume=...)  │
   ├──────────────────────────────────┤
   │ "approved" → task_execution      │
   │    create_task(proposal, op_id)  │
   │    UNIQUE(operation_id) ⇒ no     │
   │    duplicates on double clicks,  │
   │    refreshes, re-resumes         │
   │ anything else → rejected branch  │
   └──────────────────────────────────┘
```

Interrupt rules followed (per LangGraph documentation): `interrupt()` is not
wrapped in try/except; no side effects happen before it inside the node; the
payload is JSON-serializable; ambiguous resume values are treated as
rejection. UI-level idempotency: the pending operation is resolved in the
index *before* the graph resumes, and the resume only runs if the checkpoint
is still paused at `human_approval`.

## 8. Model gateway

`ModelGateway` exposes `chat(messages, tools=None)` and
`structured(schema, messages)`. Failure policy:

| Category | Examples | Falls back? |
|---|---|---|
| `rate_limit` | HTTP 429 | yes |
| `transient` | connection, timeout, 5xx | yes |
| `authentication` | invalid key on primary | yes (different provider) |
| `model_output` | malformed/incomplete structured output | yes |
| `invalid_request` | bad schema, unknown model id | **no** — reported |
| `app_validation` | our own Pydantic/validation bugs | **no** — raised |
| `unknown` | anything unclassified | **no** — raised |

One attempt per provider (`max_retries=0` on the SDK clients); the gateway
owns the retry policy. Every call carries a run name and metadata
(component, provider, model, attempt, prompt version, attempt duration,
and — on fallback attempts — the `fallback_reason`) so LangSmith traces
are meaningful. Database writes are never retried through the model layer.

### 8.1 Ordered provider chain

The gateway holds an ordered chain of `ProviderEntry` objects built from
`LLM_PRIMARY_PROVIDER` + `LLM_FALLBACK_PROVIDERS` (+ the legacy Anthropic
fallback always appended when configured). Default:

    openrouter (Nemotron 3 Ultra) -> cloudflare (GLM-4.7-Flash) -> anthropic (Haiku 5.5)

Each provider is attempted at most once per call — the chain length is the
retry bound. Unconfigured providers stay in the chain as visible "skipped"
events. On a provider switch the SAME message list (prompt, conversation
context, retrieved RAG passages) and the SAME tool definitions are handed
to the next attempt. The Cloudflare provider is a `ChatOpenAI` instance
pointed at
`{CLOUDFLARE_BASE_URL}/{CLOUDFLARE_ACCOUNT_ID}/ai/v1` with the model
`@cf/zai-org/glm-4.7-flash`; its structured-output mode defaults to
"prompt" (JSON contract + client-side Pydantic validation + one bounded
correction), matching the free OpenRouter endpoint's capabilities.

### 8.2 Cooperative cancellation (Stop button)

Streamlit processes widget clicks only between script runs, so generation
runs in a background worker thread (`ui/generation.py`) and the script
returns to the event loop — the Stop button stays clickable. Stop sets a
per-run `threading.Event` (thread-local context, `core/cancellation.py`);
graph nodes check it at entry, inside tool loops, and immediately before
the irreversible `create_task` write, then short-circuit to
`final_response`, which emits an explicit "stopped" marker. The graph
therefore always completes its normal path — the checkpointer never sees
a half-finished superstep, no message is duplicated, and the thread stays
usable. The one limitation: the synchronous model clients cannot abort a
single in-flight HTTP request; the current call finishes (bounded by
`MODEL_TIMEOUT_SECONDS`) and its output is discarded. A monitor fragment
(`@st.fragment(run_every=0.5)`) shows the live application phase reported
by the nodes and triggers a full rerun when the job finishes.

## 9. Design decisions worth defending

- **SqliteSaver instead of MemorySaver** — memory would lose pending
  approvals on restart, breaking the E2 elective.
- **App-level `pending_operations` table** — LangGraph can resume any thread,
  but has no cheap "list all paused threads" query; the index table is a
  UI-facing complement, with the checkpointer remaining the source of truth.
- **Deterministic `task_listing`** — a status keyword parse plus a direct
  tool call is more reliable than asking a model to relay a database table.
- **Read-only tools in `general_response`** — enables contextual follow-ups
  ("what is its due date?") while keeping the write path impossible to reach.
- **Guardrail before classifier** — banned patterns are rejected with zero
  model calls, saving cost and making refusals deterministic.
