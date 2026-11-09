-- The store. `rag-trend-radar db-reset` drops the configured schema's tables and runs
-- this file; every row is rebuilt from the Enron files, so there are no migrations.
-- The radar's tables are in radar.sql, which db-reset also runs.

-- A reset drops the store's tables, never the extensions (pg_search needs pgvector in public)
-- or the embedding cache: a rebuild of the store or an experiment keeps every vector computed.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE SCHEMA IF NOT EXISTS cache;

-- Every vector ever computed, by what it depends on: the model, its pinned revision, the size
-- and the exact text. Embedding looks here first, so only text never embedded costs time.
CREATE TABLE IF NOT EXISTS cache.embeddings (
    model     text NOT NULL,
    revision  text NOT NULL,
    dims      int NOT NULL,
    text_hash text NOT NULL,
    embedding halfvec NOT NULL,
    PRIMARY KEY (model, revision, dims, text_hash)
);

-- Every rerank score ever computed. A cross-encoder reads the question and the chunk together,
-- so a score depends on both texts and the model, nothing else: a re-run scores only new pairs.
-- Readers get no grant on it: a question must not be able to plant scores for another.
CREATE TABLE IF NOT EXISTS cache.rerank_scores (
    model      text NOT NULL,
    revision   text NOT NULL,
    query_hash text NOT NULL,
    text_hash  text NOT NULL,
    score      double precision NOT NULL,
    PRIMARY KEY (model, revision, query_hash, text_hash)
);

-- Which store this is: a new id on every reset. Chunk ids are only good inside one store, since a
-- rebuild hands them out again; files kept outside it (the radar's saved weeks) record this id
-- and refuse to load against another.
CREATE TABLE store (
    id uuid NOT NULL DEFAULT gen_random_uuid()
);
INSERT INTO store DEFAULT VALUES;

-- One row per searchable thing: an email, a quoted message no email of ours holds, later an
-- arXiv paper.
-- (source, external_id) is the stable outside id: the content hash for an email, the text
-- hash for a quoted message.
CREATE TABLE documents (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source      text NOT NULL,
    external_id text NOT NULL,
    kind        text NOT NULL CHECK (kind IN ('email', 'quoted')),
    sent_at     timestamptz NOT NULL,  -- for a quoted message: when it was first quoted
    title       text NOT NULL,
    body        text NOT NULL,           -- the document's own text: what gets chunked
    quoted      text NOT NULL DEFAULT '', -- quoted replies, kept but never chunked
    public      boolean NOT NULL DEFAULT false,
    deleted_at  timestamptz,
    UNIQUE (source, external_id)
);
-- The few public documents, for the chunk policy's check (none in the Enron store).
CREATE INDEX documents_public ON documents (id) WHERE public;

-- Every file in the dataset exactly as read. Never edited: everything else rebuilds from it.
CREATE TABLE raw_files (
    path        text PRIMARY KEY,
    mailbox     text NOT NULL,
    folder      text NOT NULL,
    content     bytea NOT NULL,
    document_id bigint REFERENCES documents (id)
);

-- The replay's schedule, written once by `stage`: every file that parsed, the email it's a copy
-- of, and that email's date resolved over all its copies. Day D's emails are the rows dated D.
CREATE TABLE copies (
    path           text PRIMARY KEY REFERENCES raw_files (path),
    email_key      text NOT NULL,  -- content hash, shared by every copy of one email
    sent_at        timestamptz NOT NULL,
    date_uncertain boolean NOT NULL,
    own_hash       text NOT NULL   -- its own text's hash: a quote matching any isn't an orphan
);
CREATE INDEX copies_sent_at ON copies (sent_at);
CREATE INDEX copies_own_hash ON copies (own_hash);

-- Boilerplate as known on each day: a trailing line counts from the day the 200th distinct
-- email holding it was sent, so a day's chunks don't depend on mail that came after.
CREATE TABLE boilerplate (
    line     text PRIMARY KEY,
    from_day date NOT NULL
);

-- Files set aside: a date outside 1997–2002 (522 on the 1979 placeholder, 70 stamped 2004 when
-- CMU processed them, a few from 1986 to 2044), a date that doesn't parse, a bounce, or an
-- address-book dump. `reason` is one of those four; `detail` holds the value at fault.
CREATE TABLE quarantine (
    path   text PRIMARY KEY REFERENCES raw_files (path),
    reason text NOT NULL,
    detail text NOT NULL DEFAULT ''
);

-- The email-only details of a document.
CREATE TABLE emails (
    document_id bigint PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
    message_id  text NOT NULL,
    sender      text NOT NULL,  -- the From address, or X-From's name when the export mangled it
    to_addrs    text[] NOT NULL DEFAULT '{}',
    cc_addrs    text[] NOT NULL DEFAULT '{}',
    bcc_addrs   text[] NOT NULL DEFAULT '{}',
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
-- holding an email that quotes it. Row-level security below reads it. `from_at` is when that
-- mailbox could first read it: an email's date, or the mailbox's first quote of a quoted message,
-- so a search as of an earlier day doesn't show a mailbox what it hadn't received yet.
CREATE TABLE access (
    document_id bigint REFERENCES documents (id) ON DELETE CASCADE,
    mailbox     text REFERENCES employees (mailbox),
    from_at     timestamptz NOT NULL,
    PRIMARY KEY (document_id, mailbox)
);
-- A reader's rows, found without reading everyone's (row-level security asks this every search).
CREATE INDEX access_mailbox ON access (mailbox, document_id);

-- Which emails quote which quoted message. A quoted message's readers are recounted from it,
-- so they widen as later days quote it and shrink when a quoting email is deleted.
CREATE TABLE quotes (
    email_id  bigint REFERENCES documents (id) ON DELETE CASCADE,
    quoted_id bigint REFERENCES documents (id) ON DELETE CASCADE,
    PRIMARY KEY (email_id, quoted_id)
);
CREATE INDEX quotes_quoted_id ON quotes (quoted_id);

-- Pieces of a document small enough to embed well.
CREATE TABLE chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position    int NOT NULL,
    text        text NOT NULL,
    text_hash   text NOT NULL,
    -- How many trailing characters of `text` are the message a short reply answered, separator
    -- included. Searched and embedded, never the reply's own words: above 0 tags a short reply.
    quote_chars int NOT NULL DEFAULT 0,
    UNIQUE (document_id, position)
);

-- A text embedded once is copied to every later chunk with the same text, found by its hash.
CREATE INDEX chunks_text_hash ON chunks (text_hash);

-- BM25 keyword index (pg_search). Common words out and word endings folded: "the" matched 318k
-- chunks a question, and "employee" missed "employees". Same scores on 295 questions, hybrid
-- search 10-35% faster, index 132 -> 109 MB (2026-11-02). The same options apply to queries.
CREATE INDEX chunks_bm25 ON chunks USING bm25
    (id, (text::pdb.unicode_words('stemmer=english', 'stopwords_language=english')))
    WITH (key_field = 'id');

-- One vectors table per embedding model, created by index/vectors.py when a model is first used:
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
    chunks_embedded int NOT NULL DEFAULT 0
);

-- Row-level security. The pipeline writes as the owner (radar), which bypasses it.
-- Every search runs as radar_reader with app.person set on the connection
-- ('investigator', or an employee's mailbox). Unset means nothing is visible.
DO $$ BEGIN
    CREATE ROLE radar_reader NOLOGIN;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
GRANT radar_reader TO CURRENT_USER;
-- The schema being built: public for the store, exp_<name> for an experiment.
DO $$ BEGIN
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO radar_reader', current_schema());
END $$;

-- Readers get tables by name, never by default: a new table stays invisible until it's granted
-- here with a policy. raw_files and quarantine hold every email whole, so they're never granted.
-- index/vectors.py grants each vectors_<model> table with a policy through chunks.
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

-- A chunk is visible when its document is, checked through access directly: through documents'
-- policy, every search scanned all 328k documents to build the reader's visible set (213 ms of
-- a 765 ms keyword search, 2026-10-30); from access_mailbox it takes 4 ms. Same rows: a deleted
-- document has no chunks left, so documents' deleted_at check has nothing to add here.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY chunks_visible ON chunks FOR SELECT TO radar_reader USING (
    current_setting('app.person', true) = 'investigator'
    OR EXISTS (
        SELECT 1 FROM access a
        WHERE a.document_id = chunks.document_id
          AND a.mailbox = current_setting('app.person', true)
    )
    OR EXISTS (SELECT 1 FROM documents d WHERE d.id = chunks.document_id AND d.public)
);

-- An email's details are visible when its document is; the subquery goes through documents' policy.

ALTER TABLE emails ENABLE ROW LEVEL SECURITY;
CREATE POLICY emails_visible ON emails FOR SELECT TO radar_reader USING (
    EXISTS (SELECT 1 FROM documents WHERE documents.id = emails.document_id)
);
