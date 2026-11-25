"""Pydantic schemas: the structured-output contract of the whole application.

Every model response is validated through one of these schemas before the
application uses it (concept C1).  The schemas are also used to validate the
LLM-independent database records.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    """Supported user intents (exhaustive list).

    Follow-up task, section 5.2: the router must distinguish document
    questions from general-knowledge questions, and both from current
    date/time requests.  Follow-up task 4, section 8.3 adds the explicit
    fact-check intent: "check whether my document is correct" activates
    the dedicated fact-checking workflow (ordinary document questions do
    NOT trigger it).

    current_datetime is no longer answered by a deterministic graph node:
    the request is routed to the tool-calling agent, where the MODEL decides
    to invoke the registered get_current_datetime tool (follow-up task 4,
    section 7).
    """

    DOCUMENT_QUESTION = "document_question"
    LIST_TASKS = "list_tasks"
    CREATE_TASK = "create_task"
    GENERAL_KNOWLEDGE = "general_knowledge"
    GENERAL_CONVERSATION = "general_conversation"
    CURRENT_DATETIME = "current_datetime"
    FACT_CHECK = "fact_check"
    UNSUPPORTED_REQUEST = "unsupported_request"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# C1: structured output schemas
# ---------------------------------------------------------------------------
class IntentResult(BaseModel):
    """Result of classifying a user request."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    requires_retrieval: bool = False
    requires_tool: bool = False
    clarification_needed: bool = False

    @model_validator(mode="after")
    def _enforce_consistency(self) -> "IntentResult":
        # Deterministic consistency rules; the classifier's structural flags
        # must match the chosen intent so routing stays predictable.
        if self.intent == Intent.DOCUMENT_QUESTION:
            self.requires_retrieval = True
            self.requires_tool = True
        elif self.intent == Intent.LIST_TASKS:
            self.requires_retrieval = False
            self.requires_tool = True
        elif self.intent == Intent.CREATE_TASK:
            self.requires_retrieval = False
            self.requires_tool = True
        elif self.intent == Intent.FACT_CHECK:
            # The fact-check workflow retrieves documents and runs tools.
            self.requires_retrieval = True
            self.requires_tool = True
        else:
            self.requires_retrieval = False
        return self


class TaskProposal(BaseModel):
    """A proposed task. Must be approved by the human before it is saved."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=120)
    description: str = Field(default="", max_length=2000)
    due_date: date | None = None
    priority: TaskPriority = TaskPriority.MEDIUM
    confirmation_message: str = Field(min_length=1, max_length=500)

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Task title cannot be empty or only whitespace.")
        return cleaned

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("due_date")
    @classmethod
    def _due_date_not_in_past(cls, value: date | None) -> date | None:
        if value is not None and value < date.today():
            raise ValueError("Due date cannot be in the past.")
        return value


class Citation(BaseModel):
    """A single verifiable citation attached to a grounded answer."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    page_number: int | None = None
    section: str | None = None
    chunk_id: str
    supporting_passage: str = Field(default="", max_length=1500)
    score: float | None = None


class GroundedAnswer(BaseModel):
    """An answer that must be supported by retrieved document passages."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=4000)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    insufficient_evidence: bool = False


# ---------------------------------------------------------------------------
# Retrieval / record models (LLM-independent)
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """A chunk returned by the vector store. Source of truth for citations."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int | None = None
    section: str | None = None
    text: str
    score: float | None = None


class TaskRecord(BaseModel):
    id: str
    operation_id: str | None = None
    title: str
    description: str = ""
    due_date: str | None = None
    priority: TaskPriority
    status: TaskStatus
    created_at: str


class TaskCreationResult(BaseModel):
    task: TaskRecord
    created: bool


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: Literal["processing", "indexed", "error"]
    chunk_count: int = 0
    page_count: int | None = None
    error_message: str | None = None
    uploaded_at: str


class UserPreference(BaseModel):
    id: str
    key: str
    value: str
    created_at: str
    updated_at: str


class PendingOperation(BaseModel):
    operation_id: str
    thread_id: str
    proposal: dict
    # "cancelled" = an approved run stopped before its task write
    # (follow-up task, section 7): nothing was executed.
    status: Literal[
        "pending", "approved", "rejected", "executed", "failed", "cancelled"
    ]
    created_at: str
    resolved_at: str | None = None


class ConversationInfo(BaseModel):
    thread_id: str
    title: str | None = None
    created_at: str
    updated_at: str


class IngestionResult(BaseModel):
    document_id: str
    filename: str
    # "duplicate" means the exact same content is already indexed and the
    # upload was skipped (no re-chunking, no re-embedding, no new record).
    status: Literal["indexed", "error", "duplicate"]
    chunk_count: int = 0
    page_count: int | None = None
    error_message: str | None = None
    # Wall-clock seconds per processing stage (validate / parse / chunk /
    # embed / index / total). Measured so performance can be verified.
    timings: dict[str, float] = Field(default_factory=dict)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Follow-up task 4, section 8: document cross-referencing / fact-checking
# ---------------------------------------------------------------------------
class VerificationStatus(str, Enum):
    """The evidence-based assessment of one extracted document claim.

    Follow-up task 4, section 8.6: claims are NOT forced into true/false -
    opinions, ambiguities, and unverifiable statements get their own
    categories.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    MISLEADING = "misleading"
    OUTDATED = "outdated"
    CONTRADICTORY = "contradictory"
    UNVERIFIED = "unverified"
    OPINION = "opinion"
    AMBIGUOUS = "ambiguous"


class ExternalSource(BaseModel):
    """One independent external source actually retrieved by the app.

    URLs, titles, excerpts and dates MUST come from real search results
    (the fact-check workflow validates every cited URL against the
    retrieved set and strips anything the model invented).
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=8, max_length=2000)
    domain: str = Field(default="", max_length=200)
    excerpt: str = Field(default="", max_length=1200)
    publication_date: str | None = None
    retrieval_date: str = ""


class ExtractedClaim(BaseModel):
    """One factual claim extracted from an uploaded document.

    The original wording is preserved verbatim - a claim is never rewritten
    in a way that changes its meaning.
    """

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(min_length=3, max_length=1500)
    document_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    context: str = Field(default="", max_length=1500)


class ExtractedClaims(BaseModel):
    """Structured-output schema of the claim-extraction step."""

    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=20)

    @field_validator("claims")
    @classmethod
    def _bound_claims(cls, value: list[ExtractedClaim]) -> list[ExtractedClaim]:
        # Hard bound even if the prompt's cap was exceeded by the model.
        return value[:20]


class ClaimAssessment(BaseModel):
    """Model-knowledge comparison of one claim (pre-external-verification)."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(min_length=3, max_length=1500)
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    corrected_statement: str = Field(default="", max_length=1000)
    explanation: str = Field(default="", max_length=2000)
    # True when independent external evidence is needed to decide.
    needs_external_verification: bool = False


class ClaimAssessmentBatch(BaseModel):
    """Structured-output schema of the model-knowledge check step."""

    assessments: list[ClaimAssessment] = Field(default_factory=list, max_length=20)


class FactCheckResult(BaseModel):
    """The final verified finding for one claim (follow-up task 4, 8.7)."""

    model_config = ConfigDict(extra="forbid")

    original_claim: str = Field(min_length=3, max_length=1500)
    document_id: str | None = None
    filename: str | None = None
    page_number: int | None = None
    status: VerificationStatus
    confidence: float = Field(ge=0.0, le=1.0)
    corrected_statement: str = Field(default="", max_length=1000)
    explanation: str = Field(default="", max_length=2000)
    document_evidence: str = Field(default="", max_length=1500)
    #: "model_knowledge" | "model_knowledge_and_external" | "document_only"
    #: | "external_only" - the evidence basis of the finding, so the UI can
    #: distinguish model knowledge from independently verified conclusions.
    verification_method: str = Field(default="model_knowledge", max_length=60)
    external_sources: list[ExternalSource] = Field(default_factory=list)
    limitations: str = Field(default="", max_length=1000)


class FactCheckReport(BaseModel):
    """Structured-output schema of the final fact-check report step."""

    findings: list[FactCheckResult] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=2000)
