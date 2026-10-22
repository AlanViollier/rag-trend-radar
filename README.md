# rag-trend-radar

RAG and a trend radar over one Postgres store, fed day by day.

The Enron mailbox is replayed one simulated day at a time, as if you were the compliance team
inside the company in 2001. Every day the store takes in that day's mail. Employees can search
and ask questions about their own mail, enforced by Postgres row-level security. An
investigator sees the whole corpus: what's rising, what's new, alerts, and weekly digests.

**Status:** under construction, milestone 1 of 5. Nothing here runs end to end yet.

## Run it

```sh
cp .env.template .env            # set POSTGRES_PASSWORD
docker compose up -d             # Postgres 18 + pgvector + pg_search
uv sync --extra models           # local embedding and rerank models
uv run rag-trend-radar --help
```
