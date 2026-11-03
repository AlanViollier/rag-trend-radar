-- The radar's tables, apart from the store's: `radar-reset` recreates them in seconds and leaves
-- mail, chunks and vectors alone, so a radar experiment never costs a rebuild. Everything here
-- derives from the store. Only the investigator reads it: a topic mixes many people's mail.

-- A topic's centre is frozen at birth, so re-running a day gives the same answer. A cluster
-- moderately near an older topic is born with it as parent, which is how drift and splits show.
CREATE TABLE topics (
    id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model     text NOT NULL,
    centre    halfvec NOT NULL,
    name      text NOT NULL DEFAULT '',
    keywords  text[] NOT NULL DEFAULT '{}',
    born_on   date NOT NULL,
    parent_id bigint REFERENCES topics (id)
);

-- Which topic took a chunk, and on which simulated day. Append-only, so what was known on a day
-- is the same query with an earlier date. A chunk with no row is in the unassigned pool.
CREATE TABLE assignments (
    chunk_id    bigint PRIMARY KEY REFERENCES chunks (id) ON DELETE CASCADE,
    topic_id    bigint NOT NULL REFERENCES topics (id),
    assigned_on date NOT NULL
);
CREATE INDEX assignments_topic ON assignments (topic_id, assigned_on);

-- Documents per topic per day, recounted from `assignments`, never incremented (rule 5). Short
-- replies and bulk mail aren't counted; `documents` holds the rest, `press` the press among them.
CREATE TABLE topic_days (
    topic_id  bigint REFERENCES topics (id),
    day       date,
    documents int NOT NULL,
    press     int NOT NULL,
    PRIMARY KEY (topic_id, day)
);
CREATE INDEX topic_days_day ON topic_days (day);

-- What a day's shares divide by: every document counted that day, with a topic or without.
CREATE TABLE day_totals (
    day       date PRIMARY KEY,
    documents int NOT NULL
);

-- An investigator's saved worries, checked on each day's mail by word match and by meaning.
CREATE TABLE watchlists (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name         text NOT NULL UNIQUE,
    query        text NOT NULL,
    query_vector halfvec NOT NULL,
    threshold    double precision NOT NULL
);

-- What the radar raised on a day. `expected` is the normal level a signal compared against; a
-- birth or a watchlist match has none.
CREATE TABLE alerts (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    day          date NOT NULL,
    kind         text NOT NULL CHECK (kind IN ('spike', 'rising', 'accelerating', 'birth', 'watchlist')),
    topic_id     bigint REFERENCES topics (id),
    watchlist_id bigint REFERENCES watchlists (id),
    observed     double precision NOT NULL,
    expected     double precision,
    CHECK ((kind = 'watchlist') = (watchlist_id IS NOT NULL)),
    CHECK ((kind = 'watchlist') = (topic_id IS NULL)),
    UNIQUE NULLS NOT DISTINCT (day, kind, topic_id, watchlist_id)
);
CREATE INDEX alerts_day ON alerts (day);

GRANT SELECT ON topics, assignments, topic_days, day_totals, watchlists, alerts TO radar_reader;

ALTER TABLE topics ENABLE ROW LEVEL SECURITY;
CREATE POLICY topics_investigator ON topics FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE assignments ENABLE ROW LEVEL SECURITY;
CREATE POLICY assignments_investigator ON assignments FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE topic_days ENABLE ROW LEVEL SECURITY;
CREATE POLICY topic_days_investigator ON topic_days FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE day_totals ENABLE ROW LEVEL SECURITY;
CREATE POLICY day_totals_investigator ON day_totals FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE watchlists ENABLE ROW LEVEL SECURITY;
CREATE POLICY watchlists_investigator ON watchlists FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY alerts_investigator ON alerts FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
