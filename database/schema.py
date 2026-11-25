"""Database schema initialization.

Tables:
    tasks               -- task records (written only after human approval)
    user_memory         -- explicitly saved long-term preferences
    documents           -- ingestion bookkeeping for the document library
    conversations       -- chat threads (thread_id registry)
    pending_operations  -- approval-center bookkeeping for task proposals

Virtual tables:
    chunks_fts          -- SQLite FTS5 full-text index of every indexed chunk
                           (the lexical half of hybrid retrieval; Qdrant
                           remains the semantic half)

The LangGraph checkpoint database is a separate SQLite file managed by
``graph/persistence.py``; it is intentionally not part of this schema.
"""

from __future__ import annotations

from database.connection import Database

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id            TEXT PRIMARY KEY,
        operation_id  TEXT UNIQUE,
        title         TEXT NOT NULL,
        description   TEXT NOT NULL DEFAULT '',
        due_date      TEXT,
        priority      TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
        status        TEXT NOT NULL DEFAULT 'open'
                      CHECK (status IN ('open', 'completed', 'cancelled')),
        created_at    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at)",
    """
    CREATE TABLE IF NOT EXISTS user_memory (
        id          TEXT PRIMARY KEY,
        key         TEXT NOT NULL UNIQUE,
        value       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id    TEXT PRIMARY KEY,
        filename       TEXT NOT NULL,
        file_type      TEXT NOT NULL,
        status         TEXT NOT NULL
                       CHECK (status IN ('processing', 'indexed', 'error')),
        chunk_count    INTEGER NOT NULL DEFAULT 0,
        page_count     INTEGER,
        error_message  TEXT,
        uploaded_at    TEXT NOT NULL,
        content_hash   TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
    "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)",
    """
    CREATE TABLE IF NOT EXISTS conversations (
        thread_id   TEXT PRIMARY KEY,
        title       TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pending_operations (
        operation_id   TEXT PRIMARY KEY,
        thread_id      TEXT NOT NULL,
        proposal_json  TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'approved', 'rejected',
                                         'executed', 'failed', 'cancelled')),
        created_at     TEXT NOT NULL,
        resolved_at    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_operations(status)",
    "CREATE INDEX IF NOT EXISTS idx_pending_thread ON pending_operations(thread_id)",
    # Lexical (FTS5) half of hybrid retrieval. Kept in the same database as
    # the document bookkeeping so a single transaction can keep it in sync.
    # UNINDEXED payload columns are stored but never tokenized.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text,
        chunk_id UNINDEXED,
        document_id UNINDEXED,
        filename UNINDEXED,
        page_number UNINDEXED,
        section UNINDEXED
    )
    """,
)


def init_schema(db: Database) -> None:
    """Create all tables and indexes if they do not exist yet.

    Also applies lightweight in-place migrations (new nullable columns) so
    databases created by earlier versions of the application keep working
    without any manual step. The migration runs BEFORE the schema statements:
    the idx_documents_hash index references the migrated column and would
    fail on a legacy table otherwise.
    """
    with db.lock:
        _migrate_documents_content_hash(db)
        for statement in SCHEMA_STATEMENTS:
            db.connection.execute(statement)
        db.connection.commit()


def _migrate_documents_content_hash(db: Database) -> None:
    """Add the content_hash column to pre-existing documents tables.

    CREATE TABLE IF NOT EXISTS cannot extend an existing table, so older
    databases are migrated with ALTER TABLE. The column is nullable: legacy
    rows simply have no hash and are treated as never-before-seen content
    (re-uploading such a file creates a fresh indexed copy, which is the
    previous behaviour). Must be called while holding ``db.lock``; a missing
    documents table (fresh install) is left to SCHEMA_STATEMENTS.
    """
    columns = {
        row[1] for row in db.connection.execute("PRAGMA table_info(documents)")
    }
    if not columns:
        return  # table does not exist yet; SCHEMA_STATEMENTS creates it
    if "content_hash" not in columns:
        db.connection.execute(
            "ALTER TABLE documents ADD COLUMN content_hash TEXT"
        )
