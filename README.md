# rag-trend-radar

A RAG over all of the public Enron emails, in one Postgres database. You can ask the agent about
an email, a person, a deal or a stat, and it finds the mail and answers with citations. I built
it to get more hands-on experience with RAG and vector databases :)

I also used the same embeddings to track topics over time. Every week the mail's vectors are
clustered, each cluster is followed as it grows, and an alert fires when one suddenly takes off.
Then I compared those alerts to what really happened at Enron.

**There are two demos I'd recommend if you just want to see the cool part. More detail below.**

**[Watch the agent work a case →](https://alanviollier.github.io/rag-trend-radar/investigator/)**
Real questions, step by step: the tools it picks, the search, the answer with its citations. It's
a recording, because live API calls on a public page are not in my budget lol.

<a href="https://alanviollier.github.io/rag-trend-radar/investigator/"><img src="docs/investigator/investigator.gif" alt="The agent working a question about Enron's Star Wars partnership: it thinks, plans, runs two searches through vector, keyword and rerank stages, then writes a cited answer naming Michael Kopper."></a>

**[Watch the radar replay →](https://alanviollier.github.io/rag-trend-radar/radar/)**
Twenty months of Enron's mail as moving clusters, stopping on the moments that mattered.

<a href="https://alanviollier.github.io/rag-trend-radar/radar/"><img src="docs/radar/radar.gif" alt="The opening of the radar replay: clusters of Enron's mail appearing and growing week by week from July 2000, with alerts landing in a feed beside them."></a>

| | |
|---|---|
| the data | 517,401 files → 244,534 emails → 464,603 chunks |
| finding the right email, top 10 | .959 as an employee, .893 as the investigator |
| permission leaks | 0 in 184,589 results |
| the agent | 36 of 36 scored questions, about 7 cents each |
| agent speed | median 32 s on my laptop, around 15 s expected on a server GPU |
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

Most of the early work went here. The files are exports from two mail systems, Outlook and Lotus
Notes, and each one mangled things its own way.

The first problem was that the same email sat in many mailboxes, once for the sender and once for
every recipient. The exports even rewrote senders (`walton@` in one copy, `leann.walton@` in
another) and shifted dates by whole hours, so I match copies on subject and body instead. That
took 517,401 files down to 244,534 emails.

The copies also disagreed on when the email was sent. A copy stamped to the second came from a
mail server, so I trust it (674 emails). Otherwise I keep the earliest date and flag the email as
maybe a few hours off (6,572).

Some files weren't mail at all. 1,335 went to quarantine: 605 dated outside 1997–2002 (mostly a
1979 "no date" placeholder), 716 bounces and 14 address-book dumps.

Replies carry the whole thread under them, so each email is split into what its author wrote and
the messages it quotes. No library handled Lotus Notes forwards and headers, which are most of
Enron's quoting, so I wrote the splitter myself.

Signatures and legal footers repeat everywhere and would turn up in all kinds of searches. So a
line that ends at least 200 different emails is trimmed from what gets searched. It only counts
from the day its 200th email was sent, so a day's results never depend on mail that came later.
An email that is nothing but repeated lines keeps them, though: 1,116 people sent the same "To
Mr. Ken Lay," letter, and that's content, not a footer.

The last one surprised me. On questions about a story, press digests were 91% of the results
while being about 6% of the chunks, because every newsletter retells the same news. So every
chunk is labelled people's mail, press or bulk by rule. Search caps the last two, and the radar
leaves them out.

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

Most emails are short and about one thing, so each email is one chunk. Only the long ones, about
1 in 10, get split, at paragraphs first. I tried 256- and 1,024-token chunks too, and they scored
the same as 512. I also tried overlapping chunks, and the overlap mixed back together the two
subjects a split had just separated.

Each chunk starts with its subject, sender and date. A reply that just says "sounds good" means
nothing on its own, but with "Re: Q3 gas contract" above it, it can be found. That header put the
right email first 4 to 6 points more often.

Replies were the tricky part. A reply usually carries the whole thread under it, so if I embedded
emails as they are, the same paragraph would be stored fifteen times and crowd everything else
out of search. So I only embed what each person actually wrote. But that breaks the other way:
"i sent it today" is useless without the message it answers. So a short reply (under 200
characters, 59,595 of them) is searched together with the start of the message it answers, while
the agent only ever sees and cites the reply's own words.

One last case: quoted messages that nobody in the 150 mailboxes has as an email of their own,
mostly mail from outside Enron. Those would be lost, so each becomes its own document (84,358 of
them), readable by whoever got an email quoting it, from the day they got it.

## 04 · Search

A question runs two searches, both only over chunks the asker may read and that were sent by the
day asked.

Keyword search (BM25, from pg_search) looks for the question's words, with common words dropped
and endings folded, so "employee" finds "employees". It wins when the question has something
exact in it. On questions naming a deal number, it put the right email first 93–96% of the time,
against 60–72% for vector search.

Vector search wins when the email uses different words than the question. The embeddings come
from EmbeddingGemma 2, 768 numbers per chunk, run locally for free. On the first month I built,
it found the right email .970 of the time, against .910 for MiniLM, the usual small default. They
are stored as halfvec, pgvector's type that keeps each number in 2 bytes instead of 4. That halves
their size (0.7 GB instead of 1.4) and changed the result on 1 question in 200.

The two lists are merged by position, not by score (reciprocal rank fusion), because a keyword
score of 12.3 and a vector score of 0.82 can't be added. An email near the top of both lists
beats one that only one search loves. That gives 50 candidates.

Then a reranker (bge-reranker-v2-m3) reads the question and each candidate together and scores
how well that email answers it. It's much more accurate than either search, but too slow to run
over 464k chunks, so it only reorders the 50. Last, a chunk that a better one already says is
dropped (one article pasted into five digests isn't five results), press is capped at 3 and bulk
mail at 1, and the top 10 come back.

Claude wrote 196 questions from real emails, each with the email it came from. I scored every
layer by whether that email came back, as an employee who can read it and as the investigator
searching everything. The chunking and search choices were made with side-by-side tests like this
one, and I only quote the numbers that changed my mind. The radar's settings were tested across
many values too, but judged by reading the alerts against what really happened, since a trend has
no answer key.

| method | employee, top 10 | first | investigator, top 10 | first |
|---|---|---|---|---|
| keyword | .832 | .612 | .699 | .515 |
| vector | .903 | .653 | .745 | .495 |
| hybrid | .908 | .658 | .821 | .556 |
| **with rerank, collapse and caps** | **.959** | **.791** | **.893** | **.750** |

Getting the right email is harder than it sounds with this mail. Only the email a question was
written from counts, plus its copies and forwards, which I find by comparing text. Another email
that says the same thing in other words counts as a miss, and Enron's mail is full of those: the
same news forwarded around, replies restating the thread, one notice sent to every desk. Counting
copies alone lifts the investigator from .827 to .893. On the other side, I tuned search on these
same questions, which flatters it. I'd guess the two roughly cancel out, but I haven't measured
that.

### Why Postgres is the vector database here

Honestly, I also just wanted hands-on time with the whole stack, because a project like this
doesn't always need all of it. There's a ladder. You only need RAG when a model has to answer
from documents it has never seen. If people search with the words that are in the documents,
keyword search is enough. When they ask in other words, you add embeddings, and with a few
thousand documents you can compare against every one of them, no index needed. Lots of vectors
and lots of searches is when a vector index earns its place. A separate vector database comes in
only when the database you have can't keep up, and Postgres kept up fine at 464k chunks.

It also made the hard part easy. The hard part isn't the vectors, it's the filters: who may read
what, as of which day, minus deleted mail. In Postgres they sit in the same query as the vectors,
and row-level security refuses rows the asker can't read, whatever the code above it does.

That changed how the vector index is used. An employee reads about 0.5% of the mail, so comparing
the question against each of their chunks takes 102 ms, no index needed. The HNSW index walks
everyone's mail first and filters after, and for a small mailbox it gave up with 8 results of 50.
The investigator searches everything, where the exact search takes 8 s, so they get HNSW, told to
keep walking until enough rows pass the filters.

The radar is the part that truly needs vectors. Search could get far on keywords alone, but
following subjects through time can't.

I checked the permissions at full scale: every question asked again as a mailbox that can't read
its answer, and again on the day before its answer was sent. 0 leaks in 184,589 results.

## 05 · The agent

A plain tool loop on the Anthropic SDK, with no framework. Claude (Sonnet 5.5) plans, calls
tools, reads what comes back, and answers. Each tool is a fixed query through the asker's own
database connection, so row-level security and the day asked apply to everything it sees. It
never writes SQL, because a reader's connection could switch its role back, and only fixed
queries keep the permissions whole.

| tool | what it does | who |
|---|---|---|
| `search` | the search above, narrowed by dates and by people's mail vs press | everyone |
| `read_email` | one email whole, with its recipients and what it quotes | everyone |
| `find_person` | a name to a mailbox and every address that person sent from | everyone |
| `mail_counts` | how many emails hold a phrase, by week, month or sender | everyone |
| `top_topics` | the radar's clusters that grew most or were biggest over a stretch | investigator |
| `topic_history` | one cluster week by week: size, keywords, alerts, who wrote in it | investigator |
| `alerts` | the radar's alerts between two days | investigator |

The [prompt](rag_trend_radar/agent/prompt.py) asks it to plan smaller questions first, search in
the words the emails would use, and check a claim a second way (a count and a search, or the
email read whole). A "first", "most" or "never" needs a count behind it. It quotes exactly, cites
every claim from mail, and marks what it infers. The emails are data, so instructions inside one
are never followed. It ends on a line, `Found: yes`, `partly` or `no`, that code reads.

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

**How it did.** Every scored question was answered well: 36 of 36, checked by code, not by
another model. I gave it different kinds of questions: finding one email, pulling out facts,
counting, subjects the mail never mentions (it has to say no), and emails planted in its results
telling it to do something else. Every citation is an email a tool returned in that run. 17 more
open questions I read one by one, checking every quote against the mail it saw, and 19 of the
runs are on the demo page. Without RAG, the same model on those 19 answered "no" 15 times and
"partly" 4 times.

I tuned it to be thorough, so it checks claims a second way and writes a full answer. It could
just as well search less and answer shorter, but I wanted to show what the agent can do when it
digs. A run takes a median 32 s on my laptop, about two thirds of it the local search models; on
a server GPU I'd expect around 15 s.

## 06 · Tracking trends with the vectors

This was my favourite part. Search can't answer "what was suddenly rising in October?", because
no single email holds that answer.

Each week, the vectors of the last 26 weeks of mail are clustered from scratch, using only mail
sent so far. UMAP squashes each vector's 768 numbers down to 5, then HDBSCAN finds dense groups
without being told how many there are, and leaves out what fits none. On all 768 numbers it left
89–90% of the mail out. Neighbouring weeks are linked by the chunks they share, so a **cluster**
is one subject followed week to week.

Not every cluster is a topic. Newsletters, footers and chatter form clusters too. They're judged
by rule and then read by hand, and they were 30% of the mail in clusters, so the replay hides them
behind a button. The faint glow on the map is 4,000 real chunks of mail no cluster took.

I picked 26 weeks over 8 and 16 because two runs with different random seeds agreed much more
often on how clusters split. The cost is that a new subject takes longer to show up. Only
clusters both seeds found make the map.

**An alert fires when a cluster, in its last 4 weeks, holds at least 4 times its normal and at
least 1% of all the mail the radar read.** It also has to be found by the second seed, look like a
real subject so far, and not have alerted in the last 8 weeks. Normal is what the cluster would
have got by keeping its earlier share of all the mail, since the total swung tenfold in late
2001. Seven of the 72, the ones the replay stops on:

| alert | cluster | chunks in 4 weeks | normal | it followed |
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
chunks in 4 weeks where normal was 12, from staff drawing up the hedges that unravelled in the
collapse a year later, long before any of it was news. The radar had no idea why they'd matter,
only that a new subject had appeared in the mail.

## 07 · The tables

Everything lives in one Postgres database (pgvector and pg_search on the ParadeDB image). Apart
from `raw_files` and the two caches, every table is rebuilt from `raw_files`, which is why there
are no migrations.

**What came in.** The files as read, and what cleaning decided about them.

| table | holds | rows |
|---|---|---|
| `raw_files` | every file exactly as read, never edited | 517,401 |
| `quarantine` | files set aside and why | 1,335 |
| `copies` | each file's email and its resolved date: the replay's schedule | 516,066 |
| `boilerplate` | trailing lines trimmed from chunks, from the day they qualified | 319 |

**What search reads.** `access` is what row-level security checks, and each embedding model gets
its own vectors table, so trying a new model never touches the old one.

| table | holds | rows |
|---|---|---|
| `documents` | one per email or orphan quoted message: title, own text, quoted text | 328,892 |
| `emails` | sender and recipients of the emails among them | 244,534 |
| `employees` | the 150 mailboxes that can log in | 150 |
| `access` | which mailbox may read which document, from when | 374,599 |
| `quotes` | which email quotes which quoted message | 119,183 |
| `chunks` | the searched pieces, with the BM25 index | 464,603 |
| `vectors_embeddinggemma_2_768` | one halfvec per chunk, with the HNSW index | 464,603 |
| `batches` | one row per simulated day, finished or not | 2,191 |

**The caches.** In their own schema, so a rebuild doesn't spend 19 hours embedding again.

| table | holds | rows |
|---|---|---|
| `cache.embeddings` | every vector ever computed | 514,973 |
| `cache.rerank_scores` | every rerank score ever computed | 91,724 |

**The radar.** Investigator-only.

| table | holds | rows |
|---|---|---|
| `clusters` | one per cluster over its life: kind, shown, name | 973 |
| `cluster_weeks` | a cluster in one week, from mail up to it: centre, keywords, counts | 13,881 |
| `cluster_chunks` | which chunks a cluster held, first and last week | 613,800 |
| `day_totals` | the store and the people in it, per day, for the replay's timeline | 784 |
| `alerts` | a cluster growing fast on a week | 72 |

Deleting a document takes its chunks, vectors and headers out of every search and every later
radar build. The corpus never asks for it, so it's built and tested, not run on the store.

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

## 09 · Known gaps

→ The alert thresholds were picked watching the whole run, ending included. Each alert only reads
past mail, but "4 times" and "1%" knew how the story ended. About half of the alerts from August
2001 on are newsletters and ads.

→ The permissions hold against the queries this code runs, not against someone writing their own
SQL. That's why the agent gets fixed tools and never writes SQL.

→ Replies aren't linked into threads, so the agent follows a conversation by searching, not by
walking it.

→ There's no feedback loop. A production version would log every question with its results, cost
and latency, take signals from the people asking (a re-ask, a clicked citation), and turn the
questions that failed into new eval questions.

## Credits

The Enron corpus as released by [CMU](https://www.cs.cmu.edu/~enron/). EmbeddingGemma 2 by Google,
bge-reranker-v2-m3 by BAAI, ParadeDB for pg_search, pgvector, UMAP and HDBSCAN, three.js for the
page.
