# rag-trend-radar

RAG and a trend radar over one Postgres store, fed day by day.

The Enron mailbox is replayed one simulated day at a time, as if you were the compliance team
inside the company in 2001. Every day the store takes in that day's mail. Employees can search
and ask questions about their own mail, enforced by Postgres row-level security. An
investigator sees the whole corpus: what's rising, what's new, and alerts when a subject grows fast.

**Status:** milestone 3 of 5. The whole corpus is in, replayed day by day: 244,534 emails and
84,358 quoted messages, 464,603 chunks. Questions get hybrid search, a reranker and answers from
Claude that cite the emails they come from. The radar is built and its page comes next; the agent comes with milestone 4.

## Ask it something

```sh
$ uv run rag-trend-radar ask "How did the September 11 attacks affect Enron?" --as-of 2001-09-30
investigator, as of 2001-09-30

Based on the news clippings in your mail, the attacks mainly caused a short trading and
office shutdown at Enron, followed by a quick return to business. […]

[1] Enron Mentions (document 2132)
[2] Enron Mentions - 09/15/01 - 09/16/01 (document 3342)
```

Ask as an employee and the search only sees their mailbox; ask as of an earlier day and it
only sees mail sent by then:

```sh
$ uv run rag-trend-radar ask "Who wanted to sell gas to Silicon Valley Power?" --person kean-s --as-of 2001-09-30
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
carry a BM25 index (pg_search), without common words and with word endings folded. A reply under
200 characters is indexed with the start of the message it answered: "i sent it today" can't
otherwise be found by the judge named in the message above it. Results and citations keep the
reply's own words apart.

→ **Permissions.** An employee reads what sat in their mailbox. Every search runs as a
database role whose rows are filtered by row-level security, before anything reaches the
model. A canary test plants a secret in one mailbox and searches for it as others, by every
path, in CI. A full check over 630 questions found 0 leaks in 185,589 results, by permission or
by date.

→ **Search.** Keyword and vector results (50 each) are merged by reciprocal rank fusion in
one SQL query, then a cross-encoder (bge-reranker-v2-m3) reorders them. An employee's vector
search measures every chunk they can read, since an index walk through everyone's mail runs out
before finding enough of theirs; the investigator, who reads everything, uses HNSW.

→ **The 10 that reach Claude.** A chunk a better result already says drops out, so one article
pasted into five digests counts once. Press digests take at most 3 of the 10 and bulk mail 1,
unless little else is left: on story questions they were 91% of results, from about 6% of the
chunks, and the mail people wrote rarely surfaced. They go to Claude Sonnet 5.5 as search-result
blocks, so each claim comes back with its citation.

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
day that email came in. A mailbox reads a quoted message only from its own first quote, so a
search as of an earlier day doesn't show what hadn't arrived yet. One caveat: dates, and the check
for whether any email holds a quoted text, use the whole 2002 snapshot, so day D knows copies a
live pipeline would see later.

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

Questions written by Claude from real emails, scored by whether the right email comes back in the
top 10 and first (`recall`), over the whole corpus. A forward or re-send of the right email counts
as right, by comparing text, never by a model.

| method | employee, top 10 | first | investigator, top 10 | first |
|---|---|---|---|---|
| keyword (BM25) | .832 | .612 | .699 | .515 |
| vector (EmbeddingGemma 2, 768) | .903 | .653 | .745 | .495 |
| hybrid | .908 | .658 | .821 | .556 |
| **what `ask` uses** | **.959** | **.791** | **.893** | **.750** |

196 questions, each asked two ways: as an employee who can read the answer, among a median 5,600
chunks, and as the investigator, among all 464,603. The second is the real test, and most of what
it misses is buried among mail on the same topic. On 99 questions quoting a deal or contract
number, keyword search gets the right email first 93-96% of the time and vector search 60-72%,
which is why hybrid stays; `ask` finds all 99 in its top 10.

These questions come from emails of 200 to 4,000 characters, so they can't test press digests or
short replies. Other checks do:

→ **Hard questions** (`facts`). 30 questions from the Enron story, each listing facts that must
appear in the 10 chunks an answer is written from, some only from people's own mail, plus checks
that nothing from later dates comes back. 26 pass with the plain top 10, 29 with what `ask` sends.
They were written after reading the evidence, so they're easier than blind questions.

→ **Reading the results.** Across 12 story questions, press fell from 109 of 120 slots to 61. Asked
about the Raptors, the plain top 10 is nine press digests and one email; what `ask` sends adds
Kaminski's memo refusing to back the LJM valuations and a forwarded copy of the Watkins letter.
By hand, about 33 of the 58 mail slots are people or internal documents, up from about 10; the rest
are mostly forwarded articles the rule can't detect.

→ **Short replies.** 22 questions written from a random sample of replies under 200 characters: the
investigator finds 21 in the top 10, against 14 before replies were indexed with what they answered.
The questions were chosen and written knowing the answers, so the gain is real but flattered.
The sets above didn't move.

→ **Real answers.** 10 questions through Claude, read by hand, 29 cited facts checked against the
cited emails by code, all present. It said "not in your mail" for trading strategies the mail
doesn't hold and for Skilling's resignation asked the day before it happened, and laid out where
Enron's notices and press reports disagreed on the retirement plan freeze dates. Ten answers is a
spot check, not a score.

A reranker score can't say "not in your mail" by itself: a topic the mail doesn't hold scored .29,
a real answer .06. A new question takes about 3.5 seconds before Claude starts, 1 of search and
2.5 of reranking on a laptop.

Bugs surfaced on the way, mostly where the vector index meets row-level security. An HNSW
scan fetches its nearest candidates first and the permission filter runs after, so an
employee who can read 2% of the mail got no vector results at all; iterative scans fixed it
(and the canary test now guards it). Then measuring each layer alone showed HNSW finding the
right email in the top 50 for 87% of questions where exact search found it for 100%: most
candidates it looked at were filtered out. Widening the search (`ef_search` 1,000) closed the
gap at the same 38 ms a query. The final answers barely moved through either fix, because the
merged list and the reranker covered for the broken layer, which is why each layer gets
measured on its own. At full scale the same layer-by-layer scoring found the index's scan limit
returning as few as 8 of 50 results for a small mailbox, which is what moved employees to exact
search: vector top 50 from .908 to .944, in less time. Halfvec storage costs one question in 200 against full precision
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
uv run rag-trend-radar replay --since 1997-01-01 --until 2002-12-31   # every day in order, embedding as it goes (~19 h on an M1 Pro; September 2001 alone, the default, ~30 min)
uv run rag-trend-radar ask "…"
uv run rag-trend-radar recall --person investigator   # score search, free
uv run rag-trend-radar facts       # hard questions, free
uv run rag-trend-radar leaks evals/questions_full.jsonl   # permission and date checks, free
```

The dataset goes in `data/enron/maildir` ([CMU, May 2015](https://www.cs.cmu.edu/~enron/)).
