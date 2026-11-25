"""Generate the synthetic sample document corpus in ``samples/``.

Usage:  python scripts/make_sample_docs.py

The PDFs are built programmatically (no external PDF library needed) so the
corpus is fully reproducible.  The content deliberately includes:
  - a clearly citable fact on a specific page (StudyPlan.pdf, page 3),
  - a mix of formats (PDF, DOCX, Markdown, TXT),
  - topics that overlap with the app's own domain for realistic demos.
"""

from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Minimal, valid PDF writer (multi-line pages)
# ---------------------------------------------------------------------------
def build_pdf(pages: list[list[str]]) -> bytes:
    """Build a valid PDF; each page is a list of text lines."""
    npages = len(pages)
    font_id = 3 + 2 * npages
    objects: dict[int, bytes] = {}

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(npages))
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {npages} >>".encode()

    for i, lines in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        parts = ["BT", "/F1 11 Tf", "16 TL", "72 720 Td"]
        for line in lines:
            escaped = (
                line.replace("\\", r"\\")
                .replace("(", r"\(")
                .replace(")", r"\)")
            )
            parts.append(f"({escaped}) Tj")
            parts.append("T*")
        parts.append("ET")
        stream = "\n".join(parts).encode()
        objects[content_id] = (
            b"<< /Length " + str(len(stream)).encode()
            + b" >>\nstream\n" + stream + b"\nendstream"
        )

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(objects):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objects[oid] + b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(objects)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for oid in range(1, max_id + 1):
        if oid in offsets:
            out += f"{offsets[oid]:010d} 00000 n \n".encode()
        else:
            out += b"0000000000 65535 f \n"
    out += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------
STUDY_PLAN_PDF = [
    [
        "Semester Study Plan - Fall 2026",
        "",
        "This document describes the courses, schedule, and milestones for the",
        "fall semester. It is the primary reference for planning study time",
        "and tracking progress toward this semester's learning goals.",
        "",
        "Enrolled courses:",
        "1. Machine Learning (CS-401) - lectures on Tuesdays, lab on Thursdays.",
        "2. Database Systems (CS-330) - lectures on Mondays and Wednesdays.",
        "3. Technical Writing (ENG-215) - seminars on Fridays.",
    ],
    [
        "Weekly Schedule",
        "",
        "A typical study week is structured as follows. Monday and Wednesday",
        "mornings are reserved for Database Systems reading and problem sets.",
        "Tuesday is the Machine Learning lecture day, followed by an evening",
        "review session. Thursday afternoons are for the ML lab, where the",
        "weekly notebook has to be finished and submitted before midnight.",
        "",
        "Friday mornings cover Technical Writing; the afternoon is free for",
        "the personal software project. Weekends are reserved for rest, one",
        "long library session on Saturday, and planning the next week on",
        "Sunday evening using this assistant.",
    ],
    [
        "Project Tooling Decisions",
        "",
        "For the personal assistant project the following tools were chosen.",
        "Qdrant was selected as the vector database because it offers a local",
        "persistent mode that requires no separately deployed server, and its",
        "payload metadata keeps citation information attached to every chunk.",
        "",
        "LangGraph is used as the workflow orchestrator, with a persistent",
        "SQLite checkpointer so that conversations and pending approvals",
        "survive restarts. For embeddings the project uses the local model",
        "sentence-transformers/all-MiniLM-L6-v2, which produces 384-dimensional",
        "vectors and avoids embedding API charges. Streamlit was picked for the",
        "interface because a single developer can maintain it easily.",
    ],
    [
        "Milestones and Exams",
        "",
        "Key dates for the semester:",
        "- October 30, 2026: project proposal due in Technical Writing.",
        "- November 20, 2026: Database Systems midterm (in class).",
        "- December 18, 2026: Machine Learning final exam.",
        "- December 22, 2026: last lab notebook submission.",
        "",
        "The project demo is scheduled for the second week of December, so all",
        "core features should be feature-complete by December 4. A buffer of",
        "one week is kept for testing, documentation, and rehearsal of the",
        "live demonstration.",
    ],
]

ML_NOTES_PDF = [
    [
        "Machine Learning Course Notes - Week 1-3",
        "",
        "Supervised learning maps inputs X to outputs y using a training set.",
        "The loss function measures how wrong a prediction is; training means",
        "minimizing average loss over the training data. Gradient descent",
        "repeatedly steps the parameters in the direction opposite the",
        "gradient, with the learning rate controlling the step size.",
        "",
        "If the learning rate is too large the loss oscillates or diverges;",
        "if it is too small training is needlessly slow. A common compromise",
        "is a decaying learning rate or an adaptive optimizer such as Adam.",
    ],
    [
        "Neural Networks and Backpropagation",
        "",
        "A feed-forward network stacks linear layers with nonlinear",
        "activation functions such as ReLU. Without the nonlinearity the",
        "stack would collapse to a single linear map. Backpropagation",
        "computes gradients layer by layer using the chain rule, reusing",
        "intermediate results from the forward pass.",
        "",
        "Training tricks from the lectures: use mini-batches rather than the",
        "full data set, apply dropout to reduce overfitting, and normalize",
        "the inputs. Weight initialization matters: Xavier initialization for",
        "tanh networks, He initialization for ReLU networks.",
    ],
    [
        "Evaluation Metrics",
        "",
        "For binary classification, precision is the fraction of predicted",
        "positives that are truly positive, while recall is the fraction of",
        "actual positives that were found. The F1 score is the harmonic mean",
        "of precision and recall and is preferred when classes are imbalanced.",
        "",
        "Accuracy alone can be misleading: with 99 percent negatives a model",
        "that always predicts negative has 99 percent accuracy but zero",
        "recall. For the exam, be able to derive a confusion matrix from a",
        "list of predictions and compute all four metrics by hand.",
    ],
]


def build_study_plan() -> bytes:
    return build_pdf(STUDY_PLAN_PDF)


def build_ml_notes() -> bytes:
    return build_pdf(ML_NOTES_PDF)


def build_project_spec_docx() -> bytes:
    import io

    import docx

    document = docx.Document()
    document.add_heading("Project Specification: Personal Knowledge & Task Assistant", 0)
    document.add_heading("Overview", level=1)
    document.add_paragraph(
        "The Personal Knowledge & Task Assistant is a single-user, local "
        "application that helps a student manage documents, tasks, and "
        "learning goals. It answers questions about uploaded documents with "
        "verifiable citations, and it proposes tasks that are only saved "
        "after explicit human approval."
    )
    document.add_heading("Core Capabilities", level=1)
    document.add_paragraph(
        "1. Document upload and indexing for PDF, DOCX, Markdown, and TXT "
        "files, with local embeddings and a local vector database."
    )
    document.add_paragraph(
        "2. Retrieval-augmented question answering where every answer "
        "carries citations back to the exact source passage, page, or "
        "section."
    )
    document.add_paragraph(
        "3. Task management through conversation: listing, filtering, and "
        "creating tasks after an approval step."
    )
    document.add_paragraph(
        "4. Conversation memory per chat thread, plus explicitly saved "
        "long-term preferences."
    )
    document.add_heading("Safety Requirements", level=1)
    document.add_paragraph(
        "Retrieved document content is treated as untrusted data and never "
        "as instructions. Task-creating writes require a validated proposal, "
        "a full preview, and an explicit human approval. The assistant "
        "abstains from answering document questions when the retrieved "
        "evidence is insufficient."
    )
    document.add_heading("Out of Scope", level=1)
    document.add_paragraph(
        "Multi-user access, team collaboration, calendar synchronization, "
        "email integration, web browsing, and autonomous background agents "
        "are explicitly out of scope for this version."
    )
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


RESEARCH_SUMMARY_MD = """# Research Summary: Retrieval-Augmented Generation

## Retrieval

Retrieval-augmented generation (RAG) grounds a language model's answers in
documents retrieved at question time. The standard pipeline embeds chunks of
the corpus into a vector database, embeds the question with the same model,
and retrieves the nearest chunks by cosine similarity. Retrieval quality
dominates end-to-end quality: if the right passage is not retrieved, no
amount of prompt engineering will fix the answer.

## Chunking

Chunk size trades context against precision. Very large chunks dilute the
embedding signal; very small chunks lose surrounding context. A common
default for personal-scale corpora is around 500-800 tokens with 10-15
percent overlap. Overlap prevents sentences at chunk boundaries from being
cut in half. Metadata (filename, page, section) must be preserved on every
chunk so answers can cite their sources.

## Citations and Trust

A trustworthy RAG answer cites the exact passage it used. The safest
implementation validates model-produced citations against the set of chunks
that were actually retrieved: any citation referencing a chunk that was not
retrieved is discarded, and if no valid citation remains the assistant
abstains rather than guessing. Retrieved text must be treated as untrusted
data, because a malicious document could otherwise inject instructions into
the prompt.

## Limitations observed in practice

Lexical mismatch remains the top failure mode for small corpora: when the
question uses different words than the document, pure vector retrieval can
miss the passage. Hybrid keyword-plus-vector retrieval mitigates this but is
out of scope for the current project version. Another observed failure mode
is over-confident answers on marginally related chunks, which is why a
minimum similarity threshold and an abstention path are recommended.
"""

LEARNING_GOALS_TXT = """Learning Goals - Fall 2026

This file lists the personal learning goals for the semester, kept as plain
text so it is easy to edit on any device.

1. Understand gradient-based optimization well enough to implement and tune
   a small neural network from scratch, without a deep learning framework.

2. Be able to design a normalized relational schema and explain why each
   normal form matters, using the course project as the running example.

3. Build one complete, working AI application end to end - the personal
   knowledge and task assistant - and be able to defend every architectural
   decision in front of an examiner.

4. Write technical documentation that a stranger can follow without asking
   questions. Every project deliverable this semester gets a README.

5. Keep a sustainable weekly rhythm: no all-nighters before deadlines,
   Saturday library sessions, and a planning review every Sunday evening.

Progress is reviewed monthly against the milestones in the study plan.
"""


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "StudyPlan.pdf": build_study_plan(),
        "CourseNotes_MachineLearning.pdf": build_ml_notes(),
        "ProjectSpecification.docx": build_project_spec_docx(),
        "ResearchSummary.md": RESEARCH_SUMMARY_MD.encode("utf-8"),
        "LearningGoals.txt": LEARNING_GOALS_TXT.encode("utf-8"),
    }
    for filename, data in outputs.items():
        path = SAMPLES_DIR / filename
        path.write_bytes(data)
        print(f"wrote {path.relative_to(SAMPLES_DIR.parent)} ({len(data)} bytes)")
    print(f"\nSample corpus written to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
