"""C6: RAG pipeline - parsing, chunking, metadata, vectors, citations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import guardrails
from core.models import Citation, GroundedAnswer, RetrievedChunk
from rag.chunking import chunk_sections, estimate_tokens
from rag.parsers import ParsedSection, ParsingError, parse_document


# ---------------------------------------------------------------------------
# Minimal programmatic PDF builder (valid xref, pypdf-readable)
# ---------------------------------------------------------------------------
def make_pdf(pages: list[str]) -> bytes:
    """Build a small valid PDF with one text line per page."""
    npages = len(pages)
    font_id = 3 + 2 * npages
    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(npages))
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {npages} >>".encode()
    )
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        escaped = (
            text.replace("\\", r"\\")
            .replace("(", r"\(")
            .replace(")", r"\)")
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects[content_id] = (
            b"<< /Length " + str(len(stream)).encode()
            + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects[font_id] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

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
# Parsers
# ---------------------------------------------------------------------------
def test_parse_pdf_extracts_pages_with_numbers():
    pdf_bytes = make_pdf(
        [
            "Study plan overview and goals for the semester.",
            "Qdrant was selected as the vector database for the project.",
            "The exam takes place on the last Friday of the month.",
        ]
    )
    result = parse_document("plan.pdf", pdf_bytes)
    assert result.page_count == 3
    assert len(result.sections) == 3
    assert [s.page_number for s in result.sections] == [1, 2, 3]
    assert "Qdrant" in result.sections[1].text


def test_parse_pdf_rejects_empty_pdf():
    with pytest.raises(ParsingError):
        parse_document("empty.pdf", make_pdf([" "]))


def test_parse_docx_extracts_headings():
    import docx
    import io

    document = docx.Document()
    document.add_heading("Project Overview", level=1)
    document.add_paragraph("The assistant uses LangGraph for orchestration.")
    document.add_heading("Database Choice", level=2)
    document.add_paragraph("Qdrant stores the vector embeddings locally.")
    buffer = io.BytesIO()
    document.save(buffer)

    result = parse_document("spec.docx", buffer.getvalue())
    assert result.page_count is None  # DOCX has no real page numbers
    sections_by_heading = {s.section: s for s in result.sections}
    assert "Project Overview" in sections_by_heading
    assert "LangGraph" in sections_by_heading["Project Overview"].text
    assert "Qdrant" in sections_by_heading["Database Choice"].text


def test_parse_markdown_extracts_headings():
    markdown = (
        "# Learning Goals\n\nMaster RAG pipelines.\n\n"
        "## Tools\n\nUse LangChain and Qdrant.\n"
    )
    result = parse_document("goals.md", markdown.encode())
    sections = {s.section: s for s in result.sections}
    assert "Learning Goals" in sections
    assert "Master RAG pipelines." in sections["Learning Goals"].text
    assert "Use LangChain and Qdrant." in sections["Tools"].text


def test_parse_txt_single_section():
    result = parse_document("notes.txt", b"Simple plain text notes.")
    assert len(result.sections) == 1
    assert result.sections[0].page_number is None
    assert result.sections[0].section is None


def test_parse_rejects_unsupported_extension():
    with pytest.raises(ParsingError):
        parse_document("archive.zip", b"PK")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_chunking_respects_approximate_size():
    long_text = " ".join(f"Sentence number {i} is here." for i in range(400))
    sections = [ParsedSection(text=long_text, page_number=1, section=None)]
    chunks = chunk_sections(sections, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    for chunk in chunks:
        # Allow overlap slack but no chunk should be wildly oversized.
        assert chunk.token_estimate <= 100 + 20 + 10


def test_chunking_overlap_present():
    long_text = " ".join(f"Distinct sentence {i}." for i in range(200))
    sections = [ParsedSection(text=long_text, page_number=2, section="Intro")]
    chunks = chunk_sections(sections, chunk_size_tokens=50, overlap_tokens=15)
    assert len(chunks) >= 2
    # The overlap tail of chunk N should start the chunk N+1.
    words_a = chunks[0].text.split()
    words_b = chunks[1].text.split()
    assert words_b[0] in words_a


def test_chunking_preserves_metadata():
    sections = [
        ParsedSection(text="Page one content about planning.", page_number=1, section=None),
        ParsedSection(text="Page two content about databases.", page_number=2, section=None),
    ]
    chunks = chunk_sections(sections, chunk_size_tokens=700, overlap_tokens=100)
    assert [c.page_number for c in chunks] == [1, 2]
    assert all(c.chunk_index == i for i, c in enumerate(chunks))


def test_estimate_tokens_approximation():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


# ---------------------------------------------------------------------------
# Ingestion + vector store
# ---------------------------------------------------------------------------
def test_ingestion_pipeline_end_to_end(ingestion, doc_repo):
    text = "Qdrant was selected as the vector database for the assistant."
    result = ingestion.ingest_file("plan.txt", text.encode())
    assert result.status == "indexed"
    assert result.chunk_count >= 1
    record = doc_repo.list_documents()[0]
    assert record.status == "indexed"
    assert record.filename == "plan.txt"
    assert record.chunk_count == result.chunk_count


def test_ingestion_records_error_status(ingestion, doc_repo):
    result = ingestion.ingest_file("bad.zip", b"not really a zip")
    assert result.status == "error"
    assert result.error_message and "Unsupported" in result.error_message
    # The rejected upload must not occupy library capacity.
    assert doc_repo.count_documents(("indexed", "processing")) == 0


def test_ingestion_respects_document_limit(ingestion, doc_repo, settings):
    for i in range(settings.max_documents):
        ingestion.ingest_file(f"doc{i}.txt", f"Document number {i} content.".encode())
    result = ingestion.ingest_file(
        "overflow.txt", b"One document too many."
    )
    assert result.status == "error"
    assert "full" in result.error_message
    assert doc_repo.count_documents(("indexed",)) == settings.max_documents


def test_vector_insertion_and_retrieval(ingestion, retriever):
    text_a = "The vector database chosen for the project is Qdrant."
    text_b = "The exam is scheduled for the last Friday of the semester."
    ingestion.ingest_file("plan.txt", text_a.encode())
    ingestion.ingest_file("notes.txt", text_b.encode())

    # Exact-text queries produce identical deterministic vectors (score 1.0).
    hits_a = retriever.search(text_a)
    assert hits_a and hits_a[0].filename == "plan.txt"
    assert hits_a[0].score == pytest.approx(1.0, abs=1e-6)
    hits_b = retriever.search(text_b)
    assert hits_b and hits_b[0].filename == "notes.txt"


def test_retrieval_metadata_preserved(ingestion, retriever):
    pdf_pages = [
        "Introduction to the study plan.",
        "Qdrant was selected as the vector database.",
    ]
    result = ingestion.ingest_file("plan.pdf", make_pdf(pdf_pages))
    assert result.status == "indexed"
    hits = retriever.search("Qdrant was selected as the vector database.")
    assert hits
    top = hits[0]
    assert top.filename == "plan.pdf"
    assert top.page_number == 2  # page-accurate citation metadata
    assert top.chunk_id  # verifiable identifier


def test_delete_document_removes_points(ingestion, retriever, doc_repo):
    text = "Only document in the library about quantum computing."
    result = ingestion.ingest_file("quantum.txt", text.encode())
    assert retriever.search(text)
    assert ingestion.delete_document(result.document_id) is True
    assert retriever.search(text) == []
    assert doc_repo.list_documents() == []


def test_retrieval_score_threshold_filters_noise(retriever, ingestion):
    text = "A very specific passage about Byzantine history."
    ingestion.ingest_file("history.txt", text.encode())
    # A completely different query gets an unrelated vector: with a high
    # threshold nothing should be returned.
    from rag.retrieval import Retriever

    strict = Retriever(retriever._store, top_k=5, min_score=0.999)
    assert strict.search("cooking recipes for pasta") == []


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------
def _chunk(chunk_id="c1", document_id="d1", filename="plan.pdf", page=3):
    return RetrievedChunk(
        chunk_id=chunk_id, document_id=document_id, filename=filename,
        page_number=page, section="Tools",
        text="Qdrant was selected as the vector database.",
    )


def test_citations_validated_against_actual_retrieval():
    chunks = [_chunk()]
    answer = GroundedAnswer(
        answer="Qdrant was selected as the vector database.",
        citations=[
            Citation(
                document_id="d1", filename="plan.pdf", page_number=99,
                chunk_id="c1", supporting_passage="Qdrant was selected",
            ),
            Citation(  # hallucinated chunk id: must be dropped
                document_id="dX", filename="ghost.pdf", chunk_id="cX",
                supporting_passage="invented",
            ),
        ],
        confidence=0.9,
    )
    validated = guardrails.validate_citations(answer, chunks)
    assert len(validated.citations) == 1
    kept = validated.citations[0]
    # Metadata is normalized from the actual retrieval result, not the model.
    assert kept.page_number == 3
    assert kept.filename == "plan.pdf"


def test_citation_without_valid_source_forces_abstention():
    """An unresolvable citation is reported as a problem, not accepted.

    New contract (follow-up task, section 4): validate_citations never
    silently converts the answer into the abstention text itself - it
    returns a ``problem`` so the graph can make ONE corrective retry and
    then decide. The citation itself is always rejected.
    """
    answer = GroundedAnswer(
        answer="Confident but unsupported claim.",
        citations=[
            Citation(document_id="dX", filename="ghost.pdf", chunk_id="cX",
                     supporting_passage="made up"),
        ],
        confidence=0.95,
    )
    validated = guardrails.validate_citations(answer, [])
    assert validated.problem == "unresolved_citations"
    assert validated.citations == []
    assert validated.answer == "Confident but unsupported claim."
    # The graph flow (see test_graph) turns a persistent problem into an
    # unverified answer or an honest abstention - never shown citations.


def test_abstention_when_model_flags_insufficient_evidence():
    answer = GroundedAnswer(
        answer="I could not find that in your documents.",
        citations=[], confidence=0.4, insufficient_evidence=True,
    )
    validated = guardrails.validate_citations(answer, [_chunk()])
    assert validated.insufficient_evidence is True
    assert validated.citations == []


def test_supporting_passage_replaced_when_not_quoting_chunk():
    chunks = [_chunk()]
    answer = GroundedAnswer(
        answer="Qdrant was selected as the vector database.",
        citations=[
            Citation(document_id="d1", filename="plan.pdf", chunk_id="c1",
                     supporting_passage="something completely different"),
        ],
        confidence=0.9,
    )
    validated = guardrails.validate_citations(answer, chunks)
    passage = validated.citations[0].supporting_passage
    assert "Qdrant was selected" in passage  # excerpt of the real chunk


def test_limit_context_truncates_to_budget():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", document_id="d", filename="f.pdf",
            page_number=i, section=None, text="x" * 500,
        )
        for i in range(10)
    ]
    limited = guardrails.limit_context(chunks, max_chars=1200)
    assert sum(len(chunk.text) for chunk in limited) <= 1200 + 500


def test_grounded_answer_rejects_overlong_answers():
    with pytest.raises(ValidationError):
        GroundedAnswer(answer="x" * 5000)
