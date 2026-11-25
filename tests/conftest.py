"""Shared pytest fixtures.

All tests run fully offline:
- fake gateway (no LLM API calls),
- deterministic fake embeddings (no model downloads),
- temporary SQLite databases and Qdrant local stores,
- temporary LangGraph checkpointers.

Running the suite never touches the user's real data directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Reject any non-JSON-serializable custom types in checkpoint state early.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings  # noqa: E402
from database.connection import Database  # noqa: E402
from database.conversations import ConversationRepository  # noqa: E402
from database.documents import DocumentRepository  # noqa: E402
from database.memory import MemoryRepository  # noqa: E402
from database.schema import init_schema  # noqa: E402
from database.tasks import TaskRepository  # noqa: E402
from graph.nodes import GraphServices  # noqa: E402
from graph.persistence import create_checkpointer  # noqa: E402
from graph.workflow import build_workflow  # noqa: E402
from rag.embeddings import DeterministicFakeEmbeddings  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402
from tests.fakes import FakeGateway  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Isolated settings pointing at a temporary data directory."""
    return Settings(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        data_dir=str(tmp_path / "data"),
        qdrant_path=str(tmp_path / "qdrant"),
        sqlite_path=str(tmp_path / "app.db"),
        checkpoint_db_path=str(tmp_path / "checkpoints.db"),
    )


@pytest.fixture
def db(settings) -> Database:
    database = Database(settings.sqlite_path)
    init_schema(database)
    yield database
    database.close()


@pytest.fixture
def task_repo(db) -> TaskRepository:
    return TaskRepository(db)


@pytest.fixture
def memory_repo(db) -> MemoryRepository:
    return MemoryRepository(db)


@pytest.fixture
def conv_repo(db) -> ConversationRepository:
    return ConversationRepository(db)


@pytest.fixture
def doc_repo(db) -> DocumentRepository:
    return DocumentRepository(db)


@pytest.fixture
def embeddings() -> DeterministicFakeEmbeddings:
    return DeterministicFakeEmbeddings(dim=64)


@pytest.fixture
def vector_store(settings, embeddings) -> VectorStore:
    store = VectorStore(settings.qdrant_path, embeddings)
    yield store
    store.close()


@pytest.fixture
def lexical(db):
    from rag.lexical_index import LexicalIndex

    return LexicalIndex(db)


@pytest.fixture
def retriever(settings, vector_store, lexical) -> Retriever:
    # Production parity: the same score threshold the application uses, so
    # hash-embedding cosine noise (~±0.15) can never surface as a spurious
    # retrieval hit in tests (only exact-text matches pass deterministically).
    return Retriever(
        vector_store,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        lexical_index=lexical,
    )


@pytest.fixture
def ingestion(doc_repo, vector_store, settings, lexical):
    from rag.ingestion import IngestionService

    return IngestionService(vector_store, doc_repo, settings, lexical_index=lexical)


@pytest.fixture
def gateway() -> FakeGateway:
    return FakeGateway()


@pytest.fixture
def services(gateway, retriever, task_repo, memory_repo, conv_repo, doc_repo, settings):
    return GraphServices(
        gateway=gateway,
        retriever=retriever,
        task_repo=task_repo,
        memory_repo=memory_repo,
        conversation_repo=conv_repo,
        settings=settings,
        doc_repo=doc_repo,
    )


@pytest.fixture
def graph(services, settings):
    checkpointer = create_checkpointer(settings.checkpoint_db_path)
    return build_workflow(services, checkpointer)


@pytest.fixture
def graph_factory(services, settings):
    """Build fresh graph instances sharing the same services and checkpoint
    database - used to simulate application restarts."""

    def _build():
        checkpointer = create_checkpointer(settings.checkpoint_db_path)
        return build_workflow(services, checkpointer)

    return _build
