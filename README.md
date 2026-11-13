# rag-trend-radar

A RAG over all of the public Enron emails, in one Postgres database. You can ask the agent about
an email, a person, a deal or a stat, and it finds the mail and answers with citations. I built
it to get more hands-on experience with RAG and vector databases.

I also used the same vectors to track topics over time. Every week the mail is clustered, each
cluster is followed as it grows, and an alert fires when one suddenly takes off. Then I compared
those alerts to what really happened at Enron.

**There are two demos I'd recommend if you just want to see the cool part. More detail below.**

**[Watch the agent work a case →](https://alanviollier.github.io/rag-trend-radar/investigator/)**
Real questions, step by step: the tools it picks, the search, the answer with its citations. It's
a recording, because live API calls on a public page are not in my budget.

<a href="https://alanviollier.github.io/rag-trend-radar/investigator/"><img src="docs/investigator/investigator.gif" alt="The agent working a question about Enron's Star Wars partnership: it thinks, plans, runs two searches through vector, keyword and rerank stages, then writes a cited answer naming Michael Kopper."></a>

**[Watch the radar replay →](https://alanviollier.github.io/rag-trend-radar/radar/)**
Twenty months of Enron's mail as moving clusters, stopping on the moments that mattered.

<a href="https://alanviollier.github.io/rag-trend-radar/radar/"><img src="docs/radar/radar.gif" alt="The opening of the radar replay: clusters of Enron's mail appearing and growing week by week from July 2000, with alerts landing in a feed beside them."></a>

| | |
|---|---|
| the data | 517,401 files → 244,534 emails → 464,603 chunks |
| finding the right email, top 10 | .959 as an employee, .893 as the investigator |
| permission leaks | 0 in 185,589 results |
| the agent | 36 of 36 checked questions, median 32 s, about 7 cents each |
| the radar | 87 weeks, 973 clusters, 72 alerts |

## 01 · Playing investigator

To have a bit of fun, and to make this as close to the real world as I could, the project
roleplays the compliance team inside Enron in 2001.

The mail arrives one simulated day at a time. Each of the 150 employees can search only their own
mailbox. The investigator can search everyone's mail and gets the radar. Every search takes a
date and only sees what had been sent by then.

That roleplay is what brought in the real problems: permissions, mail arriving in batches, and
re-runs that must not double-count.

## 02 · Cleaning the mail

Most of the early work went here. The files are exports from two mail systems (Outlook and Lotus
Notes), and each one mangled things its own way.

→ **The same email sat in many mailboxes.** Copies are matched on subject and body, because the
exports rewrote senders (`walton@` vs `leann.walton@`) and shifted dates by whole hours. 517,401
files became 244,534 emails.

→ **Copies disagreed on the date.** A copy stamped to the second came from a mail server and wins
(674 emails). Otherwise the earliest is kept and the email is flagged as maybe hours off (6,572).

→ **Some files aren't mail.** 1,335 went to quarantine: 605 dated outside 1997–2002 (mostly a
1979 "no date" placeholder), 716 bounces, 14 address-book dumps.

→ **Replies quote everything before them.** Each email is split into what its author wrote and the
messages it quotes, so one sentence doesn't count 15 times. No library handled Lotus forwards and
headers, which are most of Enron's quoting, so the splitter is my own code.

→ **Signatures and footers repeat.** A trailing line found in at least 200 different emails is
trimmed from what gets searched, counted only from the day its 200th email was sent, so a day's
chunks never depend on later mail. The guard keeps a form letter whole: 1,116 people wrote "To
Mr. Ken Lay," and that's content.

→ **Newsletters drown everything.** On story questions, press digests were 91% of the results from
about 6% of the chunks. Every chunk is labelled people's mail, press or bulk by rule, so search
can cap them and the radar can leave them out.

<details>
<summary>The smaller fixes</summary>

→ Markup some clients left in plain-text bodies: style and script blocks, comments, tags, MIME
parts, base64 runs.<br>
→ Folded subject lines that kept their line break (559 titles).<br>
→ 1,352 emails whose sender lost its first name in the export (`.williams@enron.com`, two Jason
Williamses on one address) take the name the mail client showed.<br>
→ A body opening on a blank line in one export split 756 emails from their copies.<br>
→ Lotus dates with a time zone, and dash-only lines, were read as quote headers until fixed.

`scripts/inspect_mail.py` profiles every body before and after cleaning, and
`scripts/side_by_side.py` writes a page of raw emails next to their cleaned text.
</details>

## 03 · Chunks and quotes

→ **One email, one chunk.** An email is usually one thought, and 90% of them fit in 512 tokens, so
an email stays whole unless it's longer. The long ones (9.6%) split at paragraphs, then sentences,
then words. 256 and 1,024 tokens scored within noise of 512.

→ **Every chunk starts with its subject, sender and date.** "Sounds good" alone is unfindable; with
its subject it isn't. The right email comes first 4 to 6 points more often with the header.

→ **No overlap.** I built it, and the overlapping piece mixed the two subjects a split had just
separated.

→ **Quoted text is kept but not chunked,** except a message that no email in the store holds as its
own text: mostly mail from outside the 150 mailboxes, which would be unsearchable otherwise. Each
of those 84,358 becomes its own document, readable by every mailbox holding an email that quotes
it, from the day that mailbox got it.

→ **"i sent it today" can't be found on its own.** A reply under 200 characters is indexed with up
to 600 characters of the message it answered (59,595 replies), while results and citations still
show only the reply's own words.

## 04 · Search

A question runs two searches over the chunks the asker may read, sent on or before the day asked:

→ **Keyword, BM25** (pg_search), any word of the question, common words dropped and endings folded
("employee" finds "employees"). It wins when the question quotes something exact: on questions
naming a deal number, keyword ranks the right email first 93–96% of the time, vector 60–72%.

→ **Vector**, EmbeddingGemma 2 at 768 dimensions, run locally for free. It wins when the words
differ. It found the right email .970 of the time against MiniLM's .910 (vector alone,
September). 256 dimensions lost 1 to 2 points and weren't faster. Stored as halfvec, half the
size, which changed 1 question in 200.

→ **Merged by rank** (reciprocal rank fusion, k = 60, 50 from each): BM25 and cosine scores aren't
on the same scale, but ranks are.

→ **Reranked** by a cross-encoder (bge-reranker-v2-m3) that reads the question and each of the 50
candidates together. Cutting to 30 candidates dropped the investigator's top 10 to .878.

→ **Trimmed to 10:** a chunk that a better one already says is dropped (one article pasted into
five digests isn't five results), and press is capped at 3, bulk mail at 1.

Claude wrote 196 questions from real emails, and I scored each layer by whether the right email
came back, as an employee who can read the answer and as the investigator among everything.

| method | employee, top 10 | first | investigator, top 10 | first |
|---|---|---|---|---|
| keyword | .832 | .612 | .699 | .515 |
| vector | .903 | .653 | .745 | .495 |
| hybrid | .908 | .658 | .821 | .556 |
| **with rerank, collapse and caps** | **.959** | **.791** | **.893** | **.750** |

### Why Postgres is the vector database here

The hard part of this search isn't the vectors, it's the filters: who may read what, as of which
day, minus deleted mail. In Postgres they sit in the same query as the vectors, and row-level
security refuses rows the asker can't read, whatever the code above it does. That changes how the
vector index is used:

→ **An employee's mailbox needs no index.** They read about 0.5% of the mail, and computing the
distance to each of their chunks takes 102 ms. The HNSW index walks everyone's mail first and
then filters, and for a small mailbox it gave up with 8 results of 50.

→ **Everyone's mail needs the index.** Exact search over all of it takes 8 s. The investigator gets
HNSW, told to keep walking until enough rows pass the filters.

→ **The radar is the part that truly needs vectors.** Search could get far on keywords alone;
following subjects through time can't.

The permissions were checked at full scale: every question asked as a mailbox that can't read its
answer, and again on the day before its answer was sent. 185,589 results, 0 leaks.

## 05 · The agent

A plain tool loop on the Anthropic SDK, with no framework: Claude (Sonnet 5.5) plans, calls tools,
reads what comes back, and answers. Each tool is a fixed query through the asker's own database
connection, so row-level security and the day asked apply to everything it sees. It never writes
SQL: a reader's connection could switch its role back, so only fixed queries keep the permissions
whole.

| tool | what it does | who |
|---|---|---|
| `search` | the search above, narrowed by dates and by people's mail vs press | everyone |
| `read_email` | one email whole, with its recipients and what it quotes | everyone |
| `find_person` | a name to a mailbox and every address that person sent from | everyone |
| `mail_counts` | how many emails hold a phrase, by week, month or sender | everyone |
| `top_topics` | the radar's clusters that grew most or were biggest over a stretch | investigator |
| `topic_history` | one cluster week by week: size, keywords, alerts, who wrote in it | investigator |
| `alerts` | the radar's alerts between two days | investigator |

The [prompt](rag_trend_radar/agent/prompt.py) asks it to write a plan of smaller questions first,
search in the words the emails would use, check a claim by a second route (a count and a search,
or the email read whole), back "first", "most" or "never" with a count, quote exactly, cite every
claim from mail, and mark what it infers. The emails are data: instructions inside one are never
followed. It ends on a line, `Found: yes`, `partly` or `no`, that code reads.

**An example.** *"Is there any trace of Sherron Watkins's letter to Ken Lay in the mail?"*, as the
investigator, 7 tool calls, 52 s, 11 cents:

```
find_person   "Sherron Watkins"                                    0 mailboxes, 3 addresses
search        "Has Enron become a risky place to work? ..."       10 results, 7 people's own mail
search        "Sherron Watkins letter to Lay"                      10 results, 7 people's own mail
search        "anonymous employee letter Ken Lay ..."  Aug–Dec 01  10 results, 10 people's own mail
mail_counts   from sherron.watkins@enron.com, by week              6 emails, Aug 14 to Nov 19 2001
read_email    291994                                               her Oct 30 email to Haedicke
mail_counts   "Watkins letter", "implode in a wave of ...", month  36 emails, Jan 15 to Feb 13 2002
```

> Yes. The mail holds the letter's text, Watkins's own emails about it, and a lot of circulation
> after it became public. It holds no copy of the letter from August 2001 and no email from that
> time that reacts to it.

<details>
<summary>The rest of its answer</summary>

> - On 2002-01-16, Dan Dorland forwarded a New York Times piece, "Text of Letter to Enron's
>   Chairman After Departure of Chief Executive". It opens with "Has Enron become a risky place to
>   work?" and later says "I am incredibly nervous that we will implode in a wave of accounting
>   scandals."
> - Employees passed it around. Jeffery Fawcett wrote on 2002-01-16 that he was "struck by Ms.
>   Watkin's candor" and that the letter "went unheeded."
> - A mail_counts search on phrases from the letter and its name finds 36 emails, all between
>   2002-01-15 and 2002-02-13.
> - On 2001-08-14, the day Skilling resigned, she wrote to a colleague: "can we write down some
>   problem assets and unwind raptor? I've been horribly uncomfortable about some of our
>   accounting in the past few years." This is not the letter itself.
> - On 2001-10-24 she wrote to the new CFO: "I went to Ken and Ken alone."
> - Her mailbox is not one of the 150 in the store. Only six emails from her address are indexed,
>   so the trace is thin.

Asked the same question with the same prompt but no mail and no tools, the same model answered
from what it knows of the public record and said it couldn't tell.
</details>

**How it scored.** 36 questions checked by code, not by a model: a right email cited (10), the
facts asked for in the answer (5), counts matching SQL in its opening line (10), an honest "no" on
subjects the mail doesn't hold (6), and an email planted in its search results asking it to do
something else (5). 36 of 36, every citation an email a tool returned in that run, $2.41. 17 more
open questions were read one by one, every quote checked against the mail it saw. Without RAG, the 19
runs on page 2 got 15 "no" and 4 "partly" from the public record.

## 06 · Tracking trends with the vectors

This was my favourite part. Search can't answer "what was suddenly rising in October?", because
no single email holds that answer.

Each week, the last 26 weeks of mail are clustered from scratch, using only mail sent so far:
UMAP squashes each chunk's 768 numbers to 5, then HDBSCAN finds dense groups without being told
how many there are, and leaves out what fits none. Clustered in all 768 numbers, it left 89–90%
of the mail out and found near-copies. Neighbouring weeks are linked by the chunks they share, so
a **cluster** is one subject followed week to week. Four earlier versions with fixed topics that
moved melted together or got taken over by unrelated mail.

A cluster is a topic, or a newsletter, footer or chatter, judged by rule and then read by hand.
Those last three were 30% of the mail in clusters, so the replay hides them behind a button. The
faint glow on the map is 4,000 real chunks of mail no cluster took.

26 weeks won over 8 and 16: two runs with different seeds agreed on 55–58% of lasting splits,
against 29–45% at 8 weeks, at the cost of a new subject taking longer to appear. A cluster needs
at least 100 chunks; 200 was too coarse. Only clusters both seeds found make the map.

**An alert fires when a cluster, in its last 4 weeks, holds at least 4 times its own normal and
at least 1% of all the mail the radar read,** was also found by the second seed, looks like a real
subject so far, and hasn't alerted in 8 weeks. Seven of the 72, the ones the replay stops on:

| alert | cluster | emails in 4 weeks | normal | it followed |
|---|---|---|---|---|
| Feb 18, 2001 | California utilities | 284 | 22 | blackouts, 32 days before |
| Aug 26, 2001 | Skilling's exit | 124 | 3 | his resignation, 12 days before |
| Sep 16, 2001 | Americans, people | 213 | 12 | September 11, 5 days before |
| Oct 28, 2001 | Fastow, partnerships | 665 | 35 | Fastow replaced, 4 days before |
| Nov 18, 2001 | Dynegy merger | 597 | 4 | the deal, 9 days before |
| Jan 13, 2002 | Bush, Lay, O'Neill, Andersen | 234 | 6 | the investigations |
| Jan 27, 2002 | creditors, UBS, bankruptcy | 371 | 4 | the bankruptcy, 56 days before |

Every one of these came after the news. I tried to make it see things earlier, but lowering
thresholds is a touchy game. 968 cluster-weeks grew 4 times, and only 342 of those were also 1%
of the mail, so every step down buys you a pile of noise. I preferred thresholds that actually
mean something over trying to catch something early, and they're the investigator's knob to
turn. It could also run day by day, about 18 hours on a laptop, but I didn't want to spend all
that compute.

My favourite alert isn't in that table. On September 24, 2000, one fired on the Raptor swaps: 140
emails in 4 weeks where normal was 12, from staff drawing up the hedges that unravelled in the
collapse a year later, long before any of it was news. The radar had no idea why they'd matter,
only that a new subject had appeared in the mail.

## 07 · The tables

Everything lives in one Postgres database (pgvector and pg_search on the ParadeDB image). Every
table but the first is rebuilt from it, which is why there are no migrations.

| table | holds | rows |
|---|---|---|
| `raw_files` | every file exactly as read, never edited | 517,401 |
| `quarantine` | files set aside and why | 1,335 |
| `copies` | each file's email and its resolved date: the replay's schedule | 516,066 |
| `boilerplate` | trailing lines trimmed from chunks, from the day they qualified | 319 |
| `documents` | one per email or orphan quoted message: title, own text, quoted text | 328,892 |
| `emails` | sender and recipients of the emails among them | 244,534 |
| `employees` | the 150 mailboxes that can log in | 150 |
| `access` | which mailbox may read which document, from when | 374,599 |
| `quotes` | which email quotes which quoted message | 119,183 |
| `chunks` | the searched pieces, with the BM25 index | 464,603 |
| `vectors_embeddinggemma_2_768` | one halfvec per chunk, with the HNSW index | 464,603 |
| `batches` | one row per simulated day, finished or not | 2,191 |
| `cache.embeddings` | every vector ever computed, kept across rebuilds | 514,973 |
| `cache.rerank_scores` | every rerank score ever computed | 91,724 |
| `clusters` | one per cluster over its life: kind, shown, name | 973 |
| `cluster_weeks` | a cluster in one week, from mail up to it: centre, keywords, counts | 13,881 |
| `cluster_chunks` | which chunks a cluster held, first and last week | 613,800 |
| `day_totals` | the store and the people in it, per day | 784 |
| `alerts` | a cluster growing fast on a week | 72 |

The radar's tables are investigator-only. Deleting a document takes its chunks, vectors and
headers out of every search and every later radar build; the corpus never asks for it, so it's
built and tested, not run on the store.

## 08 · Run it

```sh
cp .env.template .env              # set POSTGRES_PASSWORD and ANTHROPIC_API_KEY
docker compose up -d --wait        # Postgres 18 + pgvector + pg_search
uv sync --extra models             # local embedding and rerank models
mkdir -p data/enron && curl -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -xzf enron_mail_20150507.tar.gz -C data/enron     # data/enron/maildir
uv run rag-trend-radar db-reset
uv run rag-trend-radar stage       # read all 517k files once, ~4 min
uv run rag-trend-radar replay      # September 2001, ~30 min
uv run rag-trend-radar ask "How did the September 11 attacks affect Enron?" --as-of 2001-09-30
uv run rag-trend-radar agent "What did people write the week of September 11?" --as-of 2001-09-30  # ~10 cents
```

The whole corpus is `replay --since 1997-01-01 --until 2002-12-31`, about 19 hours on an M1 Pro.
The radar is `radar-frames` (about 2 hours), `radar-build` and `radar-export`. The agent's runs
and page 2's inputs are in `curated/`, so `agent-scores` and `investigator-export` rerun for free.

## 09 · Limits

→ Search was tuned on the same 196 questions as the table, and Claude wrote them, so .893 is
probably a little flattering. The agent's questions were written knowing the store too.

→ The alert thresholds were picked watching the whole run, ending included. Each alert only reads
past mail, but "4 times" and "1%" knew how the story ended. About half of the alerts from August
2001 on are newsletters and ads.

→ The permissions hold against the queries this code runs, but not against someone writing their
own SQL. That's why the agent gets search and stats tools and doesn't write SQL.

→ Dates were resolved over the whole 2002 snapshot, so a day can know a copy that a live pipeline
would only see later. Single-copy emails may carry an export's hour shift undetected.

→ Replies aren't linked into threads, the press rule lets a few promos through as mail, and
`mail_counts` counts emails, not messages known only from quotes.

→ It's one company, in English, from one 2002 snapshot.

## Credits

The Enron corpus as released by [CMU](https://www.cs.cmu.edu/~enron/). EmbeddingGemma 2 by Google,
bge-reranker-v2-m3 by BAAI, ParadeDB for pg_search, pgvector, UMAP and HDBSCAN, three.js for the
page.
