-- The radar's tables, apart from the store's: `radar-reset` recreates them in seconds and leaves
-- mail, chunks and vectors alone, so a radar experiment never costs a rebuild. Everything here
-- derives from the store. Only the investigator reads it: a topic mixes many people's mail.

-- One cluster over its whole life, however many weeks it lasted. `kind` and `shown` are judged with
-- hindsight, for the picture: alerts never read them. `name` and `family` come from the naming pass.
CREATE TABLE clusters (
    id         bigint PRIMARY KEY,
    model      text NOT NULL,
    first_week date NOT NULL,
    last_week  date NOT NULL,
    kind       text NOT NULL CHECK (kind IN ('topic', 'newsletter', 'footer', 'chatter')),
    shown      boolean NOT NULL,
    name       text NOT NULL DEFAULT '',
    family     text NOT NULL DEFAULT ''
);

-- A cluster in one week, seeing mail up to that week only, so a replay can trust it. `label` is its
-- number in that week's saved frame. `chunks` are its members; `mail` adds the chunks near it, and
-- `recent` and `recent_press` are the last 4 weeks of `mail`. `surge` is chunks above normal, `moving`
-- how far its recent centre sits from its slower average, `other_seed` whether the second seed found it.
CREATE TABLE cluster_weeks (
    cluster_id   bigint REFERENCES clusters (id) ON DELETE CASCADE,
    week         date,
    label        int NOT NULL,
    centre       halfvec NOT NULL,
    keywords     text[] NOT NULL,
    chunks       int NOT NULL,
    mail         int NOT NULL,
    recent       int NOT NULL,
    recent_press int NOT NULL,
    surge        double precision NOT NULL,
    moving       double precision NOT NULL,
    other_seed   boolean NOT NULL,
    PRIMARY KEY (cluster_id, week)
);
CREATE INDEX cluster_weeks_week ON cluster_weeks (week);

-- Which chunks a cluster held, first and last week: what search is limited to when a surge asks "what
-- happened here?". A deleted chunk leaves here at once; `cluster_weeks` recounts on the next build.
CREATE TABLE cluster_chunks (
    cluster_id bigint REFERENCES clusters (id) ON DELETE CASCADE,
    chunk_id   bigint REFERENCES chunks (id) ON DELETE CASCADE,
    first_week date NOT NULL,
    last_week  date NOT NULL,
    PRIMARY KEY (cluster_id, chunk_id)
);
CREATE INDEX cluster_chunks_chunk ON cluster_chunks (chunk_id);

-- The store and the people in it, per day, for the page's timeline. `radar_chunks` and `press_chunks`
-- count what the radar reads, without short replies and bulk mail.
CREATE TABLE day_totals (
    day          date PRIMARY KEY,
    documents    int NOT NULL,
    chunks       int NOT NULL,
    senders      int NOT NULL,
    mailboxes    int NOT NULL,
    radar_chunks int NOT NULL,
    press_chunks int NOT NULL
);

-- A cluster growing fast on a week (`radar/alerts.py`): `observed` is its chunks in the last 4 weeks,
-- `expected` its normal.
CREATE TABLE alerts (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    day        date NOT NULL,
    cluster_id bigint NOT NULL REFERENCES clusters (id) ON DELETE CASCADE,
    observed   double precision NOT NULL,
    expected   double precision NOT NULL,
    UNIQUE (day, cluster_id)
);
CREATE INDEX alerts_day ON alerts (day);

GRANT SELECT ON clusters, cluster_weeks, cluster_chunks, day_totals, alerts TO radar_reader;

ALTER TABLE clusters ENABLE ROW LEVEL SECURITY;
CREATE POLICY clusters_investigator ON clusters FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE cluster_weeks ENABLE ROW LEVEL SECURITY;
CREATE POLICY cluster_weeks_investigator ON cluster_weeks FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE cluster_chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY cluster_chunks_investigator ON cluster_chunks FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE day_totals ENABLE ROW LEVEL SECURITY;
CREATE POLICY day_totals_investigator ON day_totals FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY alerts_investigator ON alerts FOR SELECT TO radar_reader
    USING (current_setting('app.person', true) = 'investigator');
