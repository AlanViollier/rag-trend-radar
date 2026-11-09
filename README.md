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

<!-- GIF of the investigator goes here -->

**[Watch the radar replay →](https://alanviollier.github.io/rag-trend-radar/radar/)**
Twenty months of Enron's mail as moving clusters, stopping on the moments that mattered.

<!-- GIF of the replay goes here -->

| | |
|---|---|
| the data | 517,401 files → 244,534 emails → 464,603 chunks |
| finding the right email, top 10 | .959 as an employee, .893 as the investigator |
| permission leaks | 0 in 185,589 results |
| the radar | 87 weeks, 973 clusters, 72 alerts |
| a question | about 3.5 s before Claude starts, 1 to 3 cents |

## 01 · Playing investigator

To have a bit of fun, and to make this as close to the real world as I could, the project
roleplays the compliance team inside Enron in 2001.

The mail arrives one simulated day at a time. Each of the 150 employees can search only their own
mailbox. The investigator can search everyone's mail and gets the radar. Every search takes a
date and only sees what had been sent by then.

That roleplay is what brought in the real problems: permissions, mail arriving in batches, and
re-runs that must not double-count.

## 02 · Cleaning the mail

Most of the early work went here.

→ **The same email sat in many mailboxes.** Matching copies on subject and body took 517,401 files down to 244,534 emails.

→ **Replies quote everything before them.** Each email is split into what its author wrote and what it quotes, so one sentence doesn't count 15 times. No library handled Enron's Lotus forwards, so that part is my own code.

→ **Signatures and footers repeat.** A trailing line found in at least 200 different emails is trimmed from what gets searched.

→ **"i sent it today" can't be found on its own.** A reply under 200 characters is indexed with the start of the message it answered.

→ **Newsletters drown everything.** On story questions, press digests were 91% of the results from about 6% of the chunks. Every chunk is labelled mail, press or bulk by rule, so search can cap them and the radar can leave them out.

## 03 · The RAG

A question runs keyword search and vector search, 50 results each. They're merged, reranked and
trimmed to the best 10, and Claude answers from those with a citation for each claim. The agent
decides when to search and when to read the stats tables instead.

Claude wrote 196 questions from real emails, and I scored each layer by whether the right email
came back, as an employee who can read the answer and as the investigator among everything.

| method | employee, top 10 | first | investigator, top 10 | first |
|---|---|---|---|---|
| keyword | .832 | .612 | .699 | .515 |
| vector | .903 | .653 | .745 | .495 |
| hybrid | .908 | .658 | .821 | .556 |
| **with rerank, collapse and caps** | **.959** | **.791** | **.893** | **.750** |

**Why these choices:**

| I chose | over | because |
|---|---|---|
| one whole email per chunk, split only past 512 tokens | fixed-size chunks | an email is already one thought, and 90% of them fit. 256 and 1,024 tokens scored within a couple of points |
| subject, sender and date leading every chunk | the bare body | the right email comes first 4 to 6 points more often |
| no overlap between chunks | overlap | I built it, and it mixed the two subjects a split had just separated |
| EmbeddingGemma 2 | MiniLM, paid embedding APIs | runs locally for free, and found the right email .970 of the time against MiniLM's .910 (vector alone, September) |
| 768 dimensions | 256 | 1 to 2 points better on the final score, and no slower |
| halfvec | full float32 | half the storage, and it changed one question in 200 |
| keyword + vector | either alone | on questions quoting a deal number, keyword gets it first 93–96% of the time and vector 60–72%. Vector wins when the words differ |
| merging by rank (RRF) | blending scores | keyword and vector scores aren't on the same scale, but ranks are |
| bge-reranker-v2-m3, on 50 candidates | Qwen3, 30 candidates | Qwen3 was within noise. With 30 candidates the investigator's top 10 falls to .878 |
| Postgres with pgvector and pg_search | a dedicated vector database | permissions, dates, deletes and vectors sit in one query, and the filters are the hard part |
| exact search for an employee, HNSW for the investigator | one index for everyone | HNSW grabs its nearest matches first and permissions remove most of them, so a small mailbox got 8 results of 50. Exact takes 102 ms there. Over everything it takes 8 s, so the investigator gets the index |
| row-level security | filtering in the app or the prompt | the database refuses the row, whatever the code above it does |
| daily batches | streaming | most production RAG re-indexes on a schedule anyway |

## 04 · Tracking trends with the vectors

This was my favourite part. Search can't answer "what was suddenly rising in October?", because
no single email holds that answer. So each week, the last 26 weeks of mail are clustered from
scratch using only mail sent so far, and neighbouring weeks are linked by the chunks they share.
The faint glow on the map is 4,000 real chunks of mail no cluster took.

**An alert fires when a cluster, in its last 4 weeks, holds at least 4 times its own normal and
at least 1% of all the mail the radar read,** was also found by a second run with a different
seed, looks like a real subject, and hasn't alerted in 8 weeks. Seven of the 72, the ones the
replay stops on:

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
mean something over trying to catch something early. It could also run day by day, about 18 hours
on a laptop, but I didn't want to spend all that compute.

**Why these choices:**

| I chose | over | because |
|---|---|---|
| re-clustering every week | fixed topics that move | four versions of that melted together or got taken over by unrelated mail |
| UMAP, then HDBSCAN | HDBSCAN alone | alone it left 89–90% of the mail unassigned and found near-copies |
| HDBSCAN | k-means | it doesn't need the number of topics up front, and it's allowed to leave mail out |
| a 26-week window | 8 or 16 weeks | two seeds agreed on 55–58% of lasting splits, against 29–45% at 8 weeks. A new subject takes longer to appear |
| clusters of at least 100 | 200 | 200 was too coarse |
| two seeds | one | a cluster only one run finds isn't trustworthy |
| hiding chatter, footers and newsletters | showing everything | they were 30% of the mail in clusters. There's a button to bring them back |

## 05 · Run it

```sh
cp .env.template .env              # set POSTGRES_PASSWORD and ANTHROPIC_API_KEY
docker compose up -d --wait        # Postgres 18 + pgvector + pg_search
uv sync --extra models             # local embedding and rerank models
uv run rag-trend-radar db-reset
uv run rag-trend-radar stage       # read all 517k files once, ~4 min
uv run rag-trend-radar replay      # September 2001, ~30 min
uv run rag-trend-radar ask "How did the September 11 attacks affect Enron?" --as-of 2001-09-30
```

The dataset goes in `data/enron/maildir` ([CMU, May 2015](https://www.cs.cmu.edu/~enron/)). The
whole corpus takes about 19 hours on an M1 Pro, and the radar about 2 more.

## 06 · Limits

→ Search was tuned on the same 196 questions as the table, and Claude wrote them, so .893 is probably a little flattering.

→ The alert thresholds were picked watching the whole run, ending included. Each alert only reads past mail, but "4 times" and "1%" knew how the story ended. About half of the alerts from August 2001 on are newsletters and ads.

→ The permissions hold against the queries this code runs, but not against someone writing their own SQL. That's why the agent gets search and stats tools and doesn't write SQL.

→ It's one company, in English, from one 2002 snapshot.

## Credits

The Enron corpus as released by [CMU](https://www.cs.cmu.edu/~enron/). EmbeddingGemma 2 by Google,
bge-reranker-v2-m3 by BAAI, ParadeDB for pg_search, pgvector, UMAP and HDBSCAN, three.js for the
page.
