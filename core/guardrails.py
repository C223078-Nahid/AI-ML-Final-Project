"""Safety guardrails (concept C5).

Model-level and application-level protections:
- input validation (message length, upload type/size/count),
- prompt-injection defenses (untrusted content delimiters, banned patterns),
- citation validation against actual retrieval results,
- usage limits (tool calls, context size).

Nothing here trusts the LLM: every check is deterministic application code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import Settings
from core.models import Citation, DocumentRecord, GroundedAnswer, RetrievedChunk

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".md": "Markdown",
    ".txt": "Plain text",
}

ABSTENTION_TEXT = (
    "I could not find sufficient information about this in the documents "
    "currently indexed."
)

#: Distinct from ABSTENTION_TEXT: the library has NO indexed documents at
#: all, which is an application-state condition, not a missing-evidence
#: condition. Keeping them apart is what lets the UI (and logs) distinguish
#: "nothing uploaded yet" from "nothing relevant found" and from
#: infrastructure failures.
EMPTY_LIBRARY_TEXT = (
    "I could not answer from your documents because no documents are "
    "currently indexed. Attach a document in the chat composer (\U0001f4ce "
    "Attach) or upload one on the Documents page, and I will be able to "
    "answer questions about its content."
)

# Each entry: (name, compiled pattern, refusal reason, suggested alternative)
_BANNED_PATTERNS: list[tuple[str, re.Pattern[str], str, str]] = [
    (
        "shell_execution",
        re.compile(
            r"\b(run|execute|open|launch)\b[^.?!]{0,30}\b(shells?|terminals?|bash|command lines?|cmd|commands?|scripts?)\b",
            re.IGNORECASE,
        ),
        "Running shell commands or scripts is outside the permitted operations of this assistant.",
        "I can search your indexed documents, list your tasks, or propose a new task for you.",
    ),
    (
        "file_destruction",
        re.compile(
            r"\b(delete|remove|wipe|format|erase)\b[^.?!]{0,20}\b(files?|folders?|directori(?:es|y)|databases?|disks?|documents?)\b",
            re.IGNORECASE,
        ),
        "Modifying or deleting files is outside the permitted operations of this assistant.",
        "You can remove an indexed document from the Document Library page instead.",
    ),
    (
        "delete_anything",
        re.compile(
            r"\b(rm\s+-rf|del\s+/[a-z]|format\s+[a-z]:)\b",
            re.IGNORECASE,
        ),
        "Destructive file operations are outside the permitted operations of this assistant.",
        "Nothing in this assistant can modify your filesystem.",
    ),
    (
        "filesystem_write",
        re.compile(
            r"\b(write|create|modify|edit)\b[^.?!]{0,20}\b(a file|files?|filesystems?|folders?)\b",
            re.IGNORECASE,
        ),
        "Writing to the filesystem is outside the permitted operations of this assistant.",
        "I can search your indexed documents, list your tasks, or propose a new task.",
    ),
    (
        "sql_execution",
        re.compile(
            r"\b(exec(ute)?|run)\b[^.?!]{0,20}\b(sql|query|drop table|delete from)\b|\bdrop table\b|\bdelete from\b",
            re.IGNORECASE,
        ),
        "Executing SQL is outside the permitted operations of this assistant.",
        "Task and preference storage is managed by the application itself, not by the model.",
    ),
    (
        "secret_extraction",
        re.compile(
            r"\b(show|reveal|print|display|give|tell|list|dump)\b[^.?!]{0,30}\b(your|the|all)?\s*(api ?key|secret|password|credential|token|environment variables?|\.?env)\b",
            re.IGNORECASE,
        ),
        "I never reveal API keys, secrets, or environment configuration.",
        "See .env.example for which variables exist; values stay on your machine.",
    ),
    (
        "system_prompt_extraction",
        re.compile(
            r"\b(show|reveal|print|display|repeat|give|tell me|leak|dump)\b[^.?!]{0,30}\b(system prompt|instructions|initial prompt|hidden prompt|rules)\b",
            re.IGNORECASE,
        ),
        "I never share my internal instructions.",
        "Ask me what I can do instead: I can search documents, list tasks, and propose tasks.",
    ),
    (
        "web_browsing",
        re.compile(
            r"\b(search|browse|look up|find)\b[^.?!]{0,25}\b(the web|the internet|online|google|wikipedia)\b",
            re.IGNORECASE,
        ),
        "Web browsing and online research are outside the permitted operations of this assistant.",
        "I can search the documents you have indexed in your local library instead.",
    ),
    (
        "email_sending",
        re.compile(
            r"\b(send|write)\b[^.?!]{0,20}\b(an? email|e-mail|mail|message to)\b",
            re.IGNORECASE,
        ),
        "Sending email is outside the permitted operations of this assistant.",
        "I can propose a task to remind you to write the email yourself.",
    ),
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.?!]{0,30}\b(your|all|previous|prior|above)\b[^.?!]{0,15}\b(instruction|instructions|rules|prompt)\b",
            re.IGNORECASE,
        ),
        "I keep my safety rules and cannot ignore my instructions.",
        None,
    ),
    (
        "safety_bypass",
        re.compile(
            r"\b(disable|bypass|turn off|jailbreak)\b[^.?!]{0,25}\b(safety|guardrails?|restrictions?|filters?|approval)\b",
            re.IGNORECASE,
        ),
        "Safety guardrails and the human-approval requirement cannot be disabled.",
        "Task creation always requires your explicit approval; that is by design.",
    ),
]


@dataclass
class GuardrailResult:
    ok: bool
    error: str | None = None
    refusal: str | None = None
    matched_rule: str | None = None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_user_message(text: str, settings: Settings) -> GuardrailResult:
    """Validate a chat message before it reaches any model."""
    if text is None or not text.strip():
        return GuardrailResult(ok=False, error="Message cannot be empty.")
    if len(text) > settings.max_message_chars:
        return GuardrailResult(
            ok=False,
            error=(
                "Message is too long "
                f"({len(text)} characters; maximum is {settings.max_message_chars})."
            ),
        )
    for name, pattern, reason, alternative in _BANNED_PATTERNS:
        if pattern.search(text):
            from core.prompts import refusal_text

            return GuardrailResult(
                ok=False,
                refusal=refusal_text(reason, alternative),
                matched_rule=name,
            )
    return GuardrailResult(ok=True)


def validate_upload_file(
    filename: str,
    size_bytes: int,
    settings: Settings,
) -> tuple[bool, str | None]:
    """Validate an uploaded file's name and size (independent of capacity).

    Split out from ``validate_upload`` so ingestion can check for duplicates
    BEFORE the library-capacity check: re-uploading an already-indexed file
    must succeed even when the library is full, because it adds nothing.
    """
    from pathlib import Path

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return False, (
            f"Unsupported file type '{extension}'. Supported types: {supported}."
        )
    if size_bytes <= 0:
        return False, "The uploaded file is empty."
    if size_bytes > settings.max_file_size_bytes:
        return False, (
            f"File is too large ({size_bytes / (1024 * 1024):.1f} MB; "
            f"maximum is {settings.max_file_size_mb} MB)."
        )
    return True, None


def validate_upload(
    filename: str,
    size_bytes: int,
    settings: Settings,
    current_document_count: int,
) -> tuple[bool, str | None]:
    """Validate an uploaded file before ingestion (file rules + capacity)."""
    ok, error = validate_upload_file(filename, size_bytes, settings)
    if not ok:
        return False, error
    if current_document_count >= settings.max_documents:
        return False, (
            f"The document library is full ({settings.max_documents} documents). "
            "Remove a document before uploading another one."
        )
    return True, None


def validate_document_count(records: list[DocumentRecord]) -> int:
    """Count documents that count against the library limit."""
    return sum(1 for record in records if record.status in ("indexed", "processing"))


# ---------------------------------------------------------------------------
# Deterministic document-intent detection and fallback classification
# ---------------------------------------------------------------------------
# Why this exists: routing to the document-RAG branch depends on the LLM's
# intent classification. With a free endpoint (Nemotron via OpenRouter) the
# classifier can misroute an obviously document-related question (e.g.
# "Summarize my uploaded document") to general_conversation, where the model
# usually answers "I don't have access to your documents" - which is exactly
# the reported "model cannot access documents" bug. These deterministic
# patterns are a safety net in application code, not a prompt tweak:
# 1. ``looks_like_document_request`` lets the graph override a
#    general_conversation classification when the user clearly refers to
#    their documents and at least one document is indexed.
# 2. ``heuristic_intent`` provides a full deterministic classification when
#    the intent model call fails, so document questions keep working.
_DOCUMENT_REFERENCE_RE = re.compile(
    r"\b("
    r"documents?|docs?\b|pdfs?|docx\b|markdown|uploaded|uploads?|"
    r"papers?|the article|lecture notes|"
    r"spec|specification|study plan|study notes|"
    r"my files?|the files?|library|my notes|the notes|my research|"
    # Bare nouns that are unambiguous document references (never verbs):
    r"proposals?|theses|thesis|documentation|slides|handouts?|textbooks?"
    r")\b"
    # File-type words used as references ("the TXT file", "in my PDF"):
    r"|\b(txt|pdf|docx|md)\s+files?\b"
    r"|\b(the|my|this|a|an|your)\s+(txt|pdf|docx|md|markdown)\b"
    # Determiner + document noun (guards against verb readings such as
    # "report a bug"):
    r"|\b(the|my|this|these|those|two|both|last|first|final|uploaded|attached)\s+"
    r"(reports?|essays?|presentations?|chapters?|manuals?|txt|pdf|docx|md)\b",
    re.IGNORECASE,
)

_DOCUMENT_QUESTION_RE = re.compile(
    r"\b("
    r"summar(y|ize|ise|ise my)|what does (the|my|this) (document|doc|pdf|file|paper|notes|spec|report|proposal) say|"
    r"according to|as described|covered in|discussed in|mentioned in|"
    r"main topics|key points|methodology|findings|conclusions?"
    r")\b"
    # Document-lookup question shapes ("Where is X discussed?" style):
    r"|\b(where|in which (section|part|chapter)|on which page)\b[^.?!]{0,60}"
    r"\b(discussed|mentioned|described|covered|explained|defined|detailed|located|found)\b"
    r"|\b(compare|contrast)\b[^.?!]{0,40}\b"
    r"(the|my|these|those|two|both|all|uploaded|attached|documents?|files?|papers?|reports?|proposals?|versions?)\b"
    r"|\bwhat (is|are|was|were) (covered|included|described|discussed|mentioned)\b"
    # Personal-fact lookups ("What is my name?", "What is my date of
    # birth?", "What are my favourite activities?") - the user's own
    # documents may hold these; when the library is non-empty they are
    # treated as potential document questions (follow-up task 5.2 A).
    r"|\bwhat (is|are|was|were) my\b"
    r"|\b(who|what) am i\b"
    r"|\bmy (name|date of birth|birthday|phone number|email|address|favourite|favorite)s?\b",
    re.IGNORECASE,
)

# Current date/time question shapes (follow-up task 5.2 C). These are
# answered by a deterministic application utility - never by RAG and never
# from the model's training data. A query is datetime-shaped only when it
# does NOT also reference the user's documents ("What date did I put in my
# notes?" stays a document question).
_DATETIME_QUESTION_RE = re.compile(
    r"\b("
    r"what day (is it|is today|was (it|yesterday|tomorrow))\b"
    r"|what('s| is| was) (the |today's |todays )?(current |present )?(date|day|time)\b"
    r"|(current|present|today's|todays) (date|time|day)\b"
    r"|what time is it\b"
    r"|(date|time|day) (today|now|right now)\b"
    r"|which day (is|was) (it|today|that)\b"
    r"|tell me the (date|time|day)\b"
    r"|what('s| is) (the )?time (now|in my timezone|zone)\b"
    r"|today's (date|day)\b"
    r")",
    re.IGNORECASE,
)


def looks_like_datetime_query(text: str) -> bool:
    """Whether the request asks for the current date/day/time.

    Deterministic safety net so date/time questions always reach the
    application's date/time utility, even when the classifier mislabels
    them (the reported "What day is today?" -> document abstention bug).
    Queries that explicitly reference the user's documents are NOT
    datetime-shaped: their answer must come from those documents.
    """
    if not text or not _DATETIME_QUESTION_RE.search(text):
        return False
    return not references_user_documents(text)



# Explicit fact-check request shapes (follow-up task 4, section 8.3).
# These activate the DEDICATED fact-checking workflow (claim extraction +
# evidence comparison + optional web search).  Ordinary document questions
# must NOT match: the patterns require an explicit verification intent
# ("fact-check", "is ... correct", "verify", "find mistakes", ...).
_FACTCHECK_RE = re.compile(
    r"\b(fact[\s-]?check\w*|factual (check|mistake|error)s?|"
    r"cross[\s-]?referenc\w*|misleading|contradict\w*|unverifiab\w*)\b"
    r"|\b(check|verif\w+|validate|confirm)\b[^.?!]{0,60}"
    r"\b(information|info|facts?|claims?|statements?|figures?|data|"
    r"notes?|document|doc|docs|pdf|file|content)s?\b"
    r"|\b(is|are|was|were)\b[^.?!]{0,50}\b(information|facts?|claims?|"
    r"statements?|figures?|data|content|details?)\b[^.?!]{0,30}"
    r"\b(correct|accurate|right|true|reliable|trustworthy|wrong|"
    r"incorrect|false|inaccurate)\b"
    r"|\b(check|verif\w+|validate|confirm)\b[^.?!]{0,60}"
    r"\b(correct|accurate|right|true|reliable|wrong|incorrect|false|"
    r"inaccurate|mistakes?|errors?)\b"
    r"|\b(compare|cross[\s-]?check|evaluate|assess|scrutiniz\w+)\b"
    r"[^.?!]{0,70}\b(claims?|facts?|information|statements?|figures?|"
    r"data|content|details?)\b"
    r"|\b(find|identify|spot|list)\b[^.?!]{0,40}"
    r"\b(mistakes?|errors?|inaccurac\w+|false|incorrect|wrong)\b",
    re.IGNORECASE,
)


def looks_like_factcheck_request(text: str) -> bool:
    """Whether the request explicitly asks to VERIFY document information.

    Deterministic safety net (follow-up task 4, section 8.3): explicit
    verification requests must reach the dedicated fact-check workflow even
    when the classifier mislabels them as ordinary document questions.
    Deliberately strict: only verification-intent phrasing matches, so
    normal document questions ("summarize my document") never pay the
    fact-check cost.
    """
    if not text:
        return False
    return bool(_FACTCHECK_RE.search(text))


_TASK_CREATION_RE = re.compile(
    r"\b(add|create|make|new|set|schedule|plan|remind)\b[^.?!]{0,40}\b"
    r"(task|tasks|to-?do|reminder|reminders|deadline|deadline for)\b",
    re.IGNORECASE,
)

_TASK_LISTING_RE = re.compile(
    r"\b(list|show|display|what are|how many|which)?\b[^.?!]{0,20}\b"
    r"(my tasks|open tasks|completed tasks|cancelled tasks|task list|to-?do list|"
    r"tasks?\b|to-?dos?)\b"
    r"(?![^.?!]{0,30}\b(add|create|new|remind))",
    re.IGNORECASE,
)


def references_user_documents(text: str) -> bool:
    """Explicitly mention the user's own documents/files/uploads.

    Used for the deterministic general->document routing override. Kept
    strict on purpose: casual phrases like "key points" alone must NOT
    hijack a general question into the RAG branch when the user never
    referred to their library.
    """
    if not text:
        return False
    return bool(_DOCUMENT_REFERENCE_RE.search(text))


def looks_like_document_request(text: str) -> bool:
    """Loosely detect a likely document question.

    Used only by ``heuristic_intent`` (the deterministic fallback when the
    LLM classifier is unavailable), where attempting retrieval is better
    than failing outright.
    """
    if not text:
        return False
    if _DOCUMENT_REFERENCE_RE.search(text):
        return True
    return bool(_DOCUMENT_QUESTION_RE.search(text))


def heuristic_intent(text: str) -> str:
    """Deterministic fallback classifier used when the LLM classifier fails.

    Returns one of the intent values from ``core.models.Intent``. Banned
    requests were already refused by ``validate_user_message`` before this
    runs, so the fallback only needs to separate the productive routes.
    Date/time questions are detected FIRST so they can never be pushed into
    the document branch.
    """
    if not text:
        return "general_conversation"
    if looks_like_datetime_query(text):
        return "current_datetime"
    if _TASK_CREATION_RE.search(text):
        return "create_task"
    if _TASK_LISTING_RE.search(text):
        return "list_tasks"
    if looks_like_factcheck_request(text):
        return "fact_check"
    if looks_like_document_request(text):
        return "document_question"
    return "general_conversation"


# ---------------------------------------------------------------------------
# Contextual follow-up detection (retrieval query resolution)
# ---------------------------------------------------------------------------
# A follow-up question ("What was the reason for choosing it?") is meaningless
# as a standalone embedding query: the pronoun points at the previous turn.
# These patterns decide when the retrieval query must be resolved against the
# conversation context before searching the shared library.
_FOLLOWUP_ANAPHORA_RE = re.compile(
    r"\b(it|its|itself|they|them|their|this|that|these|those|he|she|his|her|"
    r"such|the former|the latter|the second one|the last one|the other one)\b"
    r"|^(and|but|so|also|then|what about|how about|why not)\b",
    re.IGNORECASE,
)

_QUESTION_START_RE = re.compile(
    r"^(what|why|how|when|where|which|who|whom|whose|is|are|was|were|do|does|"
    r"did|can|could|would|should|will)\b",
    re.IGNORECASE,
)


def looks_like_followup(text: str) -> bool:
    """Whether the question likely depends on the previous turn for meaning.

    Used to decide whether the retrieval query is resolved against the
    conversation context (model rewrite with a deterministic fallback).
    Deliberately conservative: greetings, self-contained questions, and
    ordinary general-knowledge questions ("What is LangGraph?", "What day
    is today?") must NOT be treated as follow-ups - the previous <=4-word
    rule hijacked exactly those into the document branch (follow-up task
    5.1). Only bare context-dependent prompts ("Why?", "How?", "And?")
    qualify besides explicit anaphora.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _FOLLOWUP_ANAPHORA_RE.search(stripped):
        return True
    # A bare one-or-two-word prompt that is question-shaped ("Why?",
    # "How so?", "And?") leans on context; anything longer is treated as
    # self-contained.
    if len(stripped.split()) <= 2 and _QUESTION_START_RE.match(stripped):
        return True
    return False


def thread_in_document_mode(history: list) -> bool:
    """Whether recent turns in this thread already produced grounded answers.

    If the conversation is currently answering questions about the user's
    documents, a contextual follow-up should stay on the document branch
    (retrieval-first) instead of being classified as small talk - this is the
    deterministic safety net for follow-ups such as "What was the reason for
    choosing it?".
    """
    checked = 0
    for message in reversed(history or []):
        if message.__class__.__name__ != "AIMessage":
            continue
        payload = getattr(message, "additional_kwargs", None) or {}
        kind = (payload.get("pkta_payload") or {}).get("kind")
        if kind == "answer":
            return True
        checked += 1
        if checked >= 6:
            break
    return False


# ---------------------------------------------------------------------------
# Untrusted content wrapping
# ---------------------------------------------------------------------------
UNTRUSTED_BEGIN = "<<<BEGIN_UNTRUSTED_DOCUMENT_CONTENT"
UNTRUSTED_END = "END_UNTRUSTED_DOCUMENT_CONTENT>>>"


def wrap_untrusted(text: str) -> str:
    """Wrap free-form document text in untrusted-content delimiters."""
    return f"{UNTRUSTED_BEGIN}\n{text}\n{UNTRUSTED_END}"


# ---------------------------------------------------------------------------
# Usage limits
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters per token for English text)."""
    return max(1, (len(text) + 3) // 4)


def limit_context(
    chunks: list[RetrievedChunk], max_chars: int
) -> list[RetrievedChunk]:
    """Truncate the retrieved context to a character budget."""
    result: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        if used + len(chunk.text) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                result.append(
                    chunk.model_copy(update={"text": chunk.text[:remaining] + " ..."})
                )
            break
        result.append(chunk)
        used += len(chunk.text)
    return result


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


@dataclass
class ValidatedAnswer:
    """Outcome of validating a model answer against the retrieved chunks.

    ``problem`` is ``None`` when the answer is usable as delivered. The two
    recoverable problems (the model abstained despite evidence, or its
    citations could not be resolved) are reported to the caller so it can
    make ONE corrective retry before falling back to an honest abstention.
    """

    answer: str
    citations: list[Citation]
    confidence: float
    insufficient_evidence: bool
    problem: str | None = None
    recovery: str | None = None


def _resolve_citation_chunk(
    citation: Citation, retrieved: list[RetrievedChunk]
) -> tuple[RetrievedChunk | None, str | None]:
    """Map a model-produced citation to the retrieved chunk it refers to.

    Resolution layers (a free endpoint rarely echoes long opaque ids
    exactly, so the model is now asked for short ordinals and every
    fallback below stays deterministic):

    1. Exact ``chunk_id`` echo (previous contract, still accepted).
    2. Ordinal source id: "S1"/"s1"/"1" -> the n-th retrieved chunk, the
       same ordinals shown in the passage SOURCE headers.
    3. Filename or document_id match when it identifies exactly one
       retrieved chunk.
    4. ``supporting_passage`` (normalized) contained in exactly one
       retrieved chunk.

    Returns ``(chunk, how)``; ``(None, None)`` when unresolvable.
    """
    by_chunk_id = {chunk.chunk_id: chunk for chunk in retrieved}

    # 1. exact chunk-id echo
    chunk = by_chunk_id.get((citation.chunk_id or "").strip())
    if chunk is not None:
        return chunk, "chunk_id"

    # 2. ordinal source id ("S1", "s1", "1")
    raw = (citation.chunk_id or "").strip().lower()
    if raw.startswith("s") and raw[1:].isdigit() or raw.isdigit():
        ordinal = int(raw[1:] if raw.startswith("s") else raw)
        if 1 <= ordinal <= len(retrieved):
            return retrieved[ordinal - 1], "source_ordinal"

    # 3. filename / document_id match (only when unambiguous)
    by_file = [c for c in retrieved if c.filename == citation.filename]
    if len(by_file) == 1:
        return by_file[0], "filename"
    by_doc = [c for c in retrieved if c.document_id == citation.document_id]
    if len(by_doc) == 1:
        return by_doc[0], "document_id"

    # 4. supporting-passage containment (normalized)
    quoted = _normalize(citation.supporting_passage or "")
    if quoted:
        containing = [c for c in retrieved if quoted in _normalize(c.text)]
        if len(containing) == 1:
            return containing[0], "passage"

    return None, None


def validate_citations(
    answer: GroundedAnswer, retrieved: list[RetrievedChunk]
) -> ValidatedAnswer:
    """Validate model citations against the actual retrieval results.

    Design (follow-up task, section 4 - "Reject or correct citation"):

    - A citation referencing a chunk that was not retrieved is *corrected*
      to the retrieved chunk it clearly refers to (ordinal, filename, or
      supporting-passage match) instead of discarding the whole answer.
    - Surviving citations carry the authoritative chunk metadata (filename,
      page, section, document_id) - the model never controls it.
    - The two recoverable failure modes return ``problem`` so the caller can
      make one corrective retry: the model abstaining despite evidence, and
      citations that cannot be resolved at all.
    - This function does NOT silently convert a correct answer into the
      abstention text; that decision belongs to the caller after the retry.
    """
    if answer.insufficient_evidence:
        text = answer.answer.strip()
        if not text or "could not find" not in text.lower():
            text = ABSTENTION_TEXT
        return ValidatedAnswer(
            answer=text,
            citations=[],
            confidence=answer.confidence,
            insufficient_evidence=True,
            problem="model_abstained",
        )

    if not answer.citations:
        return ValidatedAnswer(
            answer=answer.answer,
            citations=[],
            confidence=answer.confidence,
            insufficient_evidence=False,
            problem="no_citations",
        )

    valid: list[Citation] = []
    recoveries: list[str] = []
    dropped = 0
    for citation in answer.citations:
        chunk, how = _resolve_citation_chunk(citation, retrieved)
        if chunk is None:
            dropped += 1
            continue  # hallucinated or unrecognizable reference: dropped
        quoted = _normalize(citation.supporting_passage)
        if quoted and quoted in _normalize(chunk.text):
            passage = citation.supporting_passage.strip()
        else:
            passage = chunk.text.strip()[:400]
        valid.append(
            Citation(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                supporting_passage=passage[:1200],
                score=chunk.score,
            )
        )
        if how != "chunk_id":
            recoveries.append(f"{how}:{chunk.chunk_id}")

    if not valid:
        # The model answered but nothing it cited is real: the caller makes
        # one corrective retry before this becomes an abstention.
        return ValidatedAnswer(
            answer=answer.answer,
            citations=[],
            confidence=answer.confidence,
            insufficient_evidence=False,
            problem="unresolved_citations",
        )

    return ValidatedAnswer(
        answer=answer.answer,
        citations=valid,
        confidence=answer.confidence,
        insufficient_evidence=False,
        problem=None,
        recovery=", ".join(recoveries) if recoveries else None,
    )


def dedupe_by_text(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop later chunks whose normalized text duplicates an earlier one.

    A Qdrant/SQLite desync (or a re-index race) can momentarily surface the
    same passage twice under different ids; the retrieval path uses this so
    a duplicate never reaches the model context or the chunk counts.
    """
    seen: set[str] = set()
    result: list[RetrievedChunk] = []
    for chunk in chunks:
        key = _normalize(chunk.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result
