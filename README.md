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
244,541 emails. A quoted message no mailbox holds as an email becomes a document of its own.

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

## Run it

```sh
cp .env.template .env              # set POSTGRES_PASSWORD and ANTHROPIC_API_KEY
docker compose up -d --wait        # Postgres 18 + pgvector + pg_search
uv sync --extra models             # local embedding and rerank models
uv run rag-trend-radar db-reset
uv run rag-trend-radar load        # September 2001, ~2 min
uv run rag-trend-radar embed       # ~30 min on an M1 Pro
uv run rag-trend-radar ask "…"
```

The dataset goes in `data/enron/maildir` ([CMU, May 2015](https://www.cs.cmu.edu/~enron/)).
