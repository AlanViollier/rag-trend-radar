-- The store, as of milestone 1. `rag-trend-radar db-reset` drops the public schema and runs
-- this file; every row is rebuilt from the Enron files, so there are no migrations.
-- Topic, trend, watchlist and digest tables arrive with milestone 3.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- One row per searchable thing: an email, a quoted message no email of ours holds, later a
-- digest or an arXiv paper.
-- (source, external_id) is the stable outside id; a changed body_hash on it is an edit.
CREATE TABLE documents (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source      text NOT NULL,
    external_id text NOT NULL,
    kind        text NOT NULL CHECK (kind IN ('email', 'quoted', 'digest')),
    sent_at     timestamptz NOT NULL,  -- for a quoted message: when it was first quoted
    title       text NOT NULL,
    body        text NOT NULL,           -- the document's own text: what gets chunked
    quoted      text NOT NULL DEFAULT '', -- quoted replies, kept but never chunked
    body_hash   text NOT NULL,
    public      boolean NOT NULL DEFAULT false,
    deleted_at  timestamptz,
    UNIQUE (source, external_id)
);

-- Every file in the dataset exactly as read. Never edited: everything else rebuilds from it.
CREATE TABLE raw_files (
    path        text PRIMARY KEY,
    mailbox     text NOT NULL,
    folder      text NOT NULL,
    content     bytea NOT NULL,
    read_on_day date,
    document_id bigint REFERENCES documents (id)
);

-- Files whose date is junk (1979, 2044) or that don't parse, set aside with the reason.
CREATE TABLE quarantine (
    path   text PRIMARY KEY REFERENCES raw_files (path),
    reason text NOT NULL
);

-- The email-only details of a document.
CREATE TABLE emails (
    document_id bigint PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
    message_id  text NOT NULL,
    sender      text NOT NULL,
    to_addrs    text[] NOT NULL DEFAULT '{}',
    cc_addrs    text[] NOT NULL DEFAULT '{}',
    bcc_addrs   text[] NOT NULL DEFAULT '{}',
    thread_id   text,
    -- Copies disagreed on the date and none was stamped to the second: off by up to ~12 hours.
    date_uncertain boolean NOT NULL DEFAULT false
);

-- The 150 mailbox owners, the employees who can log in, named by their mailbox (kean-s).
-- A mailbox, not an address: owners write under several addresses and assistants send from
-- their boss's account, so addresses can't say whose mail it is.
CREATE TABLE employees (
    mailbox text PRIMARY KEY
);

-- Who may see which document: every mailbox a copy sat in; for a quoted message, every mailbox
-- holding an email that quotes it. Row-level security below reads it.
CREATE TABLE access (
    document_id bigint REFERENCES documents (id) ON DELETE CASCADE,
    mailbox     text REFERENCES employees (mailbox),
    PRIMARY KEY (document_id, mailbox)
);

-- Pieces of a document small enough to embed well.
CREATE TABLE chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position    int NOT NULL,
    text        text NOT NULL,
    text_hash   text NOT NULL,
    UNIQUE (document_id, position)
);

-- BM25 keyword index (pg_search).
CREATE INDEX chunks_bm25 ON chunks USING bm25 (id, text) WITH (key_field = 'id');

-- One vectors table per embedding model, created by index.py when a model is first used:
--   CREATE TABLE vectors_<model> (
--       chunk_id  bigint PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
--       embedding halfvec(<dims>) NOT NULL
--   );
--   CREATE INDEX ON vectors_<model> USING hnsw (embedding halfvec_cosine_ops);

-- The run log: one row per simulated day. as_of defaults to the last finished day.
CREATE TABLE batches (
    day             date PRIMARY KEY,
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz,
    emails_in       int NOT NULL DEFAULT 0,
    documents_new   int NOT NULL DEFAULT 0,
    chunks_embedded int NOT NULL DEFAULT 0,
    cost_usd        numeric(10, 4) NOT NULL DEFAULT 0,
    errors          int NOT NULL DEFAULT 0
);

-- Row-level security. The pipeline writes as the owner (radar), which bypasses it.
-- Every search runs as radar_reader with app.person set on the connection
-- ('investigator', or an employee's mailbox). Unset means nothing is visible.
DO $$ BEGIN
    CREATE ROLE radar_reader NOLOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
GRANT radar_reader TO CURRENT_USER;
GRANT USAGE ON SCHEMA public TO radar_reader;

-- Readers get tables by name, never by default: a new table stays invisible until it's granted
-- here with a policy. raw_files and quarantine hold every email whole, so they're never granted.
-- index.py grants each vectors_<model> table with a policy through chunks.
GRANT SELECT ON documents, emails, access, employees, chunks, batches TO radar_reader;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY documents_visible ON documents FOR SELECT TO radar_reader USING (
    deleted_at IS NULL
    AND (
        public
        OR current_setting('app.person', true) = 'investigator'
        OR EXISTS (
            SELECT 1 FROM access
            WHERE access.document_id = documents.id
              AND access.mailbox = current_setting('app.person', true)
        )
    )
);

-- Your own grants only; the investigator sees all of them.
ALTER TABLE access ENABLE ROW LEVEL SECURITY;
CREATE POLICY access_visible ON access FOR SELECT TO radar_reader USING (
    mailbox = current_setting('app.person', true)
    OR current_setting('app.person', true) = 'investigator'
);

-- A chunk or an email's details are visible when its document is; the subquery goes through
-- documents' policy.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY chunks_visible ON chunks FOR SELECT TO radar_reader USING (
    EXISTS (SELECT 1 FROM documents WHERE documents.id = chunks.document_id)
);

ALTER TABLE emails ENABLE ROW LEVEL SECURITY;
CREATE POLICY emails_visible ON emails FOR SELECT TO radar_reader USING (
    EXISTS (SELECT 1 FROM documents WHERE documents.id = emails.document_id)
);
