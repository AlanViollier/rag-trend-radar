# rag-trend-radar

RAG and a trend radar over one Postgres store, fed day by day.

The Enron mailbox is replayed one simulated day at a time, as if you were the compliance team
inside the company in 2001. Every day the store takes in that day's mail. Employees can search
and ask questions about their own mail, enforced by Postgres row-level security. An
investigator sees the whole corpus: what's rising, what's new, alerts, and weekly digests.

**Status:** milestone 1 of 5. Question answering works end to end over September 2001
(8,804 emails, 127 employees): hybrid search, a reranker, and answers from Claude that cite the
emails they come from. The daily replay, the radar and the agent come with milestones 2 to 4.

## Ask it something

```sh
$ uv run rag-trend-radar ask "How did the September 11 attacks affect Enron?"
investigator, as of 2001-09-30

Based on the news clippings in your mail, the attacks mainly caused a short trading and
office shutdown at Enron, followed by a quick return to business. […]

[1] Enron Mentions (document 2132)
[2] Enron Mentions - 09/15/01 - 09/16/01 (document 3342)
```

Ask as an employee and the search only sees their mailbox; ask as of an earlier day and it
only sees mail sent by then:

```sh
$ uv run rag-trend-radar ask "Who wanted to sell gas to Silicon Valley Power?" --person kean-s
Nothing in your search results answers this. […]

$ uv run rag-trend-radar ask "How did the September 11 attacks affect Enron?" --as-of 2001-09-10
This isn't in your mail. Today is 2001-09-10, so the September 11 attacks haven't happened yet […]
```

An answer costs one to three cents.

## How it works

→ **Ingest.** 517,401 files from the CMU Enron dataset are parsed, stripped of markup, split
into each email's own text and the replies it quotes, and deduplicated across mailboxes into
244,534 emails. A quoted message no mailbox holds as an email becomes a document of its own.

→ **Index.** Each email is one chunk, or several past 512 tokens, led by its subject, sender
and date, and embedded with EmbeddingGemma 2 into pgvector (halfvec, HNSW). The same chunks
carry a BM25 index (pg_search).

→ **Permissions.** An employee reads what sat in their mailbox. Every search runs as a
database role whose rows are filtered by row-level security, before anything reaches the
model. A canary test plants a secret in one mailbox and searches for it as others, by every
path, in CI.

→ **Search.** Keyword and vector results (50 each) are merged by reciprocal rank fusion in
one SQL query, then a cross-encoder (bge-reranker-v2-m3) reorders them and the best 10 go to
Claude Sonnet 5.5 as search-result blocks, so each claim comes back with its citation.

## The daily replay

The store is fed the way a compliance system would be: one day at a time, each day safe to run
again.

→ **Stage once.** An email's date is only known once all its copies are seen: the Lotus and
Outlook exports shift the same email by whole hours, so copies of a late-evening email can fall
on two different days. `stage` reads all 517,401 files once (4 minutes), resolves each email's
date over its copies, and writes the schedule to a table. Each day then reads its own rows,
about 3 seconds a day before embedding.

→ **Re-running a day changes nothing.** Documents are keyed by a hash of their content, and
an email is re-chunked only when its chunks would read differently; a day's counts are
recounted from the stored rows, never added to. Running all 30 days of September a second time wrote nothing. A day's row closes only once
its vectors exist, so a crashed run leaves searches on the last finished day.

→ **As known that day.** A signature line counts as boilerplate only from the day its 200th
email was sent, so a day's chunks never depend on later mail. A quoted message keeps the date
of its first quote, and its readers are recounted from every email that quotes it, whichever
day that email came in. One caveat: dates, and the check for whether any email holds a quoted
text, use the whole 2002 snapshot, so day D knows copies a live pipeline would see later.

→ **Deletes and model swaps.** `delete` takes a document out of every search on every day: its
chunks and vectors go, and the row stays as a tombstone so a replay doesn't bring it back. The
corpus never asks for one; the path is tested. A new embedding model fills its own table
beside the one serving searches, `vectors` shows how much of the corpus it covers, and the
config flips once it's full. MiniLM filled September in under two minutes, found the right
email in the top 10 for .910 of questions against EmbeddingGemma's .970, and was dropped.

Rebuilding September through the replay found a dedup bug from a refactor: a copy whose body
opened on a blank line counted as a second email (756 across the corpus). After the fix the
rebuilt store scores what the old one did, within one question in 200.

## How well it finds things

Questions written by Claude from 245 real emails and quoted messages, each asked as an employee
who can read the answer, scored by whether the right email comes back in the top 10 and first
(`recall`):

| method | 200 questions, top 10 | first | 50 with no names, top 10 | first |
|---|---|---|---|---|
| keyword (BM25) | .955 | .800 | .880 | .720 |
| vector (EmbeddingGemma 2, 768) | .970 | .795 | .960 | .760 |
| hybrid | .970 | .820 | .960 | .740 |
| **hybrid + rerank** (what `ask` uses) | **.995** | **.890** | **1.000** | **.900** |

The reranker is the largest gain: it lifts "first" from about 80% to about 90%. The no-names
set forbids naming people, companies and places, which costs keyword search and leaves vector
search where it was. With 50 questions, differences under about ten points there are noise.

Two bugs surfaced on the way, both where the vector index meets row-level security. An HNSW
scan fetches its nearest candidates first and the permission filter runs after, so an
employee who can read 2% of the mail got no vector results at all; iterative scans fixed it
(and the canary test now guards it). Then measuring each layer alone showed HNSW finding the
right email in the top 50 for 87% of questions where exact search found it for 100%: most
candidates it looked at were filtered out. Widening the search (`ef_search` 1,000) closed the
gap at the same 38 ms a query. The final answers barely moved through either fix, because the
merged list and the reranker covered for the broken layer, which is why each layer gets
measured on its own. Halfvec storage costs one question in 200 against full precision
(`scripts/precision.py`).

Two chunking choices were checked the same way, each in its own Postgres schema so the store
stays untouched. Leading every chunk with its subject, sender and date puts the right email
first 4 to 6 points more often than the bare body (hybrid .820 against .780, vector .795
against .735, 200 questions). Chunk size barely matters: 256, 512 and 1,024 tokens land within
a couple of points of each other, because most emails fit whole under all three. The Qwen3
reranker scored within noise of bge (.875 against .890 first), so bge stays.

## Run it

```sh
cp .env.template .env              # set POSTGRES_PASSWORD and ANTHROPIC_API_KEY
docker compose up -d --wait        # Postgres 18 + pgvector + pg_search
uv sync --extra models             # local embedding and rerank models
uv run rag-trend-radar db-reset
uv run rag-trend-radar stage       # read all 517k files once, ~4 min
uv run rag-trend-radar replay      # September 2001 day by day, embedding as it goes (~30 min on an M1 Pro)
uv run rag-trend-radar ask "…"
```

The dataset goes in `data/enron/maildir` ([CMU, May 2015](https://www.cs.cmu.edu/~enron/)).
