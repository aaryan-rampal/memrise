# TODO

- [ ] General: get embeddings working end-to-end
  - Current: `OpenRouterEmbedder`, embedding storage, and semantic search methods all exist (`memories/embedder.py`, `memories/sqlite_store.py`), but `mem` rejects embedding operations behind `EMBEDDINGS_DISABLED_MESSAGE` (`memories/cli.py`).
  - Next step: remove/rework the global embedding-disable guard, then validate:
    - `mem add ... --embedder openrouter`
    - `mem semantic-search ...`
    - `mem raw-chats index --embed`
  - Required check: confirm OpenRouter API key/model wiring and rate-limit behavior in one run across both curated memories and raw artifacts.

- [ ] Generate curated memory seeds from real curated sources
  - Current: `curated import` loads `data/curated/memories.jsonl`, but the pipeline does not create that JSONL; the current seed was bridged from an old SQLite backup.
  - Next step: add a generation stage that reads approved curated sources, such as Claude/Codex memory summaries, writes `data/curated/memories.jsonl`, and keeps synthesized memory generation separate from raw chat indexing.

- [ ] Investigate a co-occurrence index for raw chat retrieval
  - Current: raw chat search uses FTS over span content only (`search_raw_artifacts` in `memories/sqlite_store.py`) and no co-occurrence structure is persisted.
  - Next step: evaluate whether precomputed co-occurrence (term pairs, phrase spans, or artifact/message co-occurrence edges) improves recall for multi-term queries and noisy natural-language terms.

- [ ] Paginate raw chat search results
  - Current: search supports only `--limit` and returns the first page of results (`memories/cli.py` and `SQLiteMemoryStore.search_raw_artifacts`).
  - Next step: add offset/page parameters (and CLI flags) with deterministic ordering guarantees to avoid returning the same top-N window repeatedly.

- [ ] Make raw chat indexing incremental
  - Current: `raw-chats index` replaces indexed raw artifacts for the selected providers and rebuilds derived FTS rows, which makes reruns slower than a fresh DB rebuild.
  - Next step: detect unchanged canonical conversations and upsert only added/changed artifacts while deleting stale artifacts for the selected providers.

- [ ] Improve raw search ranking with a smarter heuristic filter
  - Current: ordering is currently raw-FTS `bm25(...)` + `span_index` in the SQL query.
  - Next step: add a reranking layer after lexical retrieval (e.g., overlap score, distinct query-term coverage, role/provider recency bias, message-window freshness) and return scored matches.

- [ ] Current-state check to remember
  - Embeddings: implemented in code paths, currently disabled in CLI dispatch and blocked in user-facing commands.
  - Curated seed generation: not present.
  - Co-occurrence index: not present.
  - Raw-chat incremental indexing: not present.
  - Raw-search pagination: not present.
  - Raw-search ranking: lexical BM25 ordering only.
