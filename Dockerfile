## ParadeDB bundles pgvector 0.8.6; 0.8.7 fixes CVE-2026-103484 (IVFFlat build overflow).
## Delete this file once a ParadeDB image ships pgvector >= 0.8.7.
FROM paradedb/paradedb:0.26.3-pg18
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-18-pgvector=0.8.7-1.pgdg13+1 \
    && rm -rf /var/lib/apt/lists/*
USER postgres
