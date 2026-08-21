"""GraphRAG retrieval connectors: the read side of the index.

GraphConnector answers "what does the graph know about this document?" (graphrag's own
local/global/basic search, plus direct entity/topology math on the parquet graph).
EmbeddingConnector does the same over the vector store. Split out of graphrag_manager, which
keeps the write side: config, indexing, corpus switching and the API server lifecycle.

Imports graphrag_manager, never the reverse - callers import the connectors from here.
OUTPUT_DIR is read as gm.OUTPUT_DIR rather than imported: set_active_corpus() rebinds it on every
corpus switch, so an import-time copy would silently pin this module to the previous corpus.
"""
from __future__ import annotations

import bisect
import json
import logging
import math
import re
import threading
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import scripts.graphrag_manager as gm
from scripts.graphrag_manager import (
    CONTEXT_CACHE_DIR,
    GRAPHRAG_NO_DATA_ANSWER,
    GRAPHRAG_ROOT,
    MAX_QUERY_CHARS,
    _context_persist_lock,
    _ingest_document,
    _run_sync,
    load_graph,
    logger,
    query,
    set_active_corpus,
)


class EmbeddingConnector:
    """Read access to one corpus's LanceDB text-unit embeddings, for detectors
    that need nearest-neighbour lookups (nli_contradiction, embedding_outlier).
    """

    # vector table name -> (source parquet, text column) to join back against.
    # graphrag's vector store rows only carry id/vector/dates, not the source
    # text, so nearest() has to look it up from the workflow output that fed it.
    # Table names on disk are "{vector_store_id}-{workflow}-{field}" (settings.yaml's
    # embed_text.vector_store_id, "default" here) - not the bare "text_unit_text" this used to
    # assume, which never actually matched any real LanceDB table (first exercised this session).
    _TEXT_SOURCES = {
        "default-text_unit-text": ("text_units.parquet", "text"),
        "default-entity-description": ("entities.parquet", "description"),
        "default-community-full_content": ("community_reports.parquet", "full_content"),
    }

    def __init__(self, corpus: str, table: str = "default-text_unit-text", embedding_model: str | None = None):
        from openai import OpenAI
        import os

        self._corpus_output = GRAPHRAG_ROOT / "output" / corpus
        self._table_name = table
        # Embeddings get their own, separate endpoint/key/model - deliberately independent from
        # OPENAI_BASE_URL/OPENAI_API_KEY (still used for completions). Reason: the shared
        # university endpoint's embedding backend proved unreliable across a full night of
        # testing (repeated silent timeouts) while its completion backend stayed solid the whole
        # time - moving only embeddings to a dedicated provider (e.g. DeepInfra, which hosts this
        # exact Qwen3-Embedding-4B model) fixes the actual broken piece without touching what
        # already works. Falls back to OPENAI_BASE_URL/OPENAI_API_KEY if the EMBEDDING_* vars
        # aren't set, so this is a no-op change until .env is actually updated.
        self._client = OpenAI(
            api_key=os.environ.get("EMBEDDING_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip(),
            base_url=os.environ.get("EMBEDDING_BASE_URL", "").strip() or os.environ.get("OPENAI_BASE_URL", "").strip(),
            # See detector.py's _llm_judge_base for why: no timeout meant the SDK's 10-minute
            # default plus its own internal retries, hidden underneath retry_api_call's own
            # retries. 60s (not the 30s first tried, which turned out to cut off normal-but-slow
            # responses on this shared endpoint - graphrag's own settings.yaml uses 180s for
            # completions) - retry_api_call still owns the retry policy on top.
            timeout=60.0, max_retries=0,
        )
        # Different providers name the same model differently (e.g. DeepInfra lists it as
        # "Qwen/Qwen3-Embedding-4B", not bare "Qwen3-Embedding-4B") - EMBEDDING_MODEL_NAME lets
        # .env pick the right string for whichever provider is actually configured.
        self._embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL_NAME", "").strip() or "Qwen3-Embedding-4B"
        self._embed_cache: dict[str, list[float]] = {}  # text -> vector, since embed() is pure and this
        # connector is reused across a whole Optuna study - without it, every trial re-embeds the exact
        # same validation docs from scratch even though only k/k_neighbors changes between trials.
        self.reload()

    def reload(self) -> None:
        """(Re-)opens the LanceDB table and refreshes the id->text lookup and distance-stats
        cache - split out of __init__ so callers can refresh in-memory state after the on-disk
        index changed underneath this connector (e.g. after a version-store restore)."""
        import lancedb

        db = lancedb.connect(str(self._corpus_output / "lancedb"))
        self._table = db.open_table(self._table_name)
        self._stats: dict[int, dict] = {}  # lazily computed corpus-wide kNN distance baseline, keyed by k

        source_file, text_column = self._TEXT_SOURCES[self._table_name]
        source_df = pd.read_parquet(self._corpus_output / source_file)
        self._id_to_text = dict(zip(source_df["id"], source_df[text_column]))

    def embed(self, text: str) -> list[float]:
        from scripts.helpers import retry_api_call
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        response = retry_api_call(self._client.embeddings.create, model=self._embedding_model, input=text)
        vector = response.data[0].embedding
        self._embed_cache[text] = vector
        return vector

    def ingest_text(self, text: str, doc_id: str | None = None) -> None:
        """Directly embeds and inserts one new row into the LanceDB table - cheap simulation of
        a document really being added, without running GraphRAG's full (graph-building) update
        pipeline, which embedding-only detectors don't need."""
        from datetime import datetime, timezone

        vector = self.embed(text)
        new_id = doc_id or f"ingest_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        self._table.add([{
            "id": new_id,
            "vector": vector,
            "create_date": now.isoformat(),
            "update_date": None,
            "create_date_year": now.year,
            "create_date_month": now.month,
            "create_date_month_name": now.strftime("%B"),
            "create_date_day": now.day,
            "create_date_day_of_week": now.strftime("%A"),
            "create_date_hour": now.hour,
            "create_date_quarter": (now.month - 1) // 3 + 1,
            "update_date_year": None,
            "update_date_month": None,
            "update_date_month_name": None,
            "update_date_day": None,
            "update_date_day_of_week": None,
            "update_date_hour": None,
            "update_date_quarter": None,
        }])
        self._id_to_text[new_id] = text

    def nearest(self, vector: list[float], k: int = 5) -> list[dict]:
        """k nearest text units to `vector`, closest first."""
        rows = self._table.search(vector).limit(k).to_list()
        return [
            {"id": r["id"], "text": self._id_to_text.get(r["id"], ""), "distance": r["_distance"]}
            for r in rows
        ]

    def corpus_distance_stats(self, k: int = 10, sample: int = 200) -> dict:
        """Mean/std of average kNN self-distance over a corpus sample, used as an
        outlier baseline so a single document's distance can be turned into a z-score.
        Keyed by k - the baseline genuinely differs per k, and this connector is reused
        across an Optuna study that searches k itself, so caching a single value here
        would silently keep serving whichever k happened to run first."""
        if k not in self._stats:
            df = self._table.to_pandas().sample(n=min(sample, self._table.count_rows()), random_state=0)
            avg_dists = [
                sum(n["distance"] for n in self.nearest(vector, k + 1)[1:]) / k
                for vector in df["vector"]
            ]
            mean = sum(avg_dists) / len(avg_dists)
            variance = sum((d - mean) ** 2 for d in avg_dists) / len(avg_dists)
            self._stats[k] = {"mean": mean, "std": variance ** 0.5}
        return self._stats[k]


class GraphConnector:
    """Read access to one corpus's entity/relationship graph, for detectors
    that need related entities/notes (grag_contradiction).
    """

    _MIN_ENTITY_TITLE_LENGTH = 4  # skip 1-3 char titles (e.g. "ER", "IL") that match as
                                  # substrings of unrelated words rather than real mentions

    def __init__(self, corpus: str):
        self.corpus = corpus
        # Retrieval-coverage tallies for evidence_coverage() - deliberately NOT reset by reload(),
        # so they accumulate across a whole detector run rather than per graph refresh.
        self._evidence_count = 0
        self._no_evidence_count = 0
        set_active_corpus(corpus)
        self.reload()

    def reload(self) -> None:
        """(Re)loads the graph and text-unit lookup from disk. Split out of __init__ so
        ingest_document() can refresh in-memory state after a real reindex without
        duplicating this logic."""
        self._graph = load_graph()
        text_units = pd.read_parquet(gm.OUTPUT_DIR / "text_units.parquet")
        self._id_to_text = dict(zip(text_units["id"], text_units["text"]))
        # precomputed once per graph load instead of on every find_entities_in_text()
        # call - this runs per document, so relowering/refiltering thousands of node
        # names each time added up fast (grag_contradiction is Optuna-tuned, meaning
        # this ran once per (trial x validation doc) too).
        self._lowered_titles = [
            (name, name.lower()) for name in self._graph.nodes if len(name) >= self._MIN_ENTITY_TITLE_LENGTH
        ]
        self._pagerank: dict[str, float] | None = None  # stale after (re)load, recomputed lazily
        self._pagerank_lock = threading.Lock()  # centrality() is called from detect_batch's
        # thread pool - see the double-checked init there for what this guards.
        self._context_cache: dict[tuple, str] = {}  # (text, method, community_level, response_type) -> response,
        # since query_context() makes real LLM calls - stale after (re)load just like _pagerank.
        self._load_persisted_contexts()
        self._community_cache: dict[int, dict[str, int]] = {}  # level -> {entity title: community id}
        self._block_cache: dict[int, dict[tuple, float]] = {}  # level -> {(community a, community b): edge rate}
        self._surprise_cache: dict[int, list[float]] = {}  # level -> sorted surprises, see pair_surprise
        self._block_model_cache: dict[int, tuple] = {}  # level -> raw block counts, see _block_model

    def _context_cache_path(self) -> Path:
        return CONTEXT_CACHE_DIR / f"{self.corpus}.jsonl"

    def _load_persisted_contexts(self) -> None:
        """Repopulates _context_cache from disk. query_context()'s results are deterministic given
        (text, method, community_level, response_type) and cost a real graphrag search each, so
        without this every process restart re-pays the whole retrieval bill - including the
        "global" method, which is by far the most expensive thing in a gnlic/grag_as_a_judge study.
        Also means n-mode's contexts are still there for dva-mode after a restart between them."""
        path = self._context_cache_path()
        if not path.exists():
            return
        loaded = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                self._context_cache[(row["text"], row["method"], row["community_level"], row["response_type"])] = row["response"]
                loaded += 1
            except Exception:
                continue  # a torn final line from an interrupted append - skip it, don't lose the rest
        if loaded:
            logger.info("Loaded %d persisted query contexts for corpus '%s'.", loaded, self.corpus)

    def _persist_context(self, key: tuple, response: str) -> None:
        """Appends one successful lookup as a JSONL row. Append-only (not a rewrite) so a crash
        mid-run keeps everything already computed, and only successes are written - a failure is
        cached in memory for this process (see query_context) but must NOT outlive it, since a
        restart is exactly when a previously-failing document deserves a fresh attempt."""
        text, method, community_level, response_type = key
        row = {"text": text, "method": method, "community_level": community_level,
               "response_type": response_type, "response": response}
        with _context_persist_lock:
            try:
                CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with self._context_cache_path().open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("Failed to persist query context - continuing with in-memory cache only.")

    def ingest_document(self, text: str) -> None:
        """Really adds `text` to the corpus via GraphRAG's incremental update pipeline, then
        reloads this connector's in-memory graph/text-unit state so it's visible immediately."""
        _ingest_document(self.corpus, text)
        self.reload()

    def query_context(self, text: str, method: str = "local", community_level: int = 2,
                       response_type: str = "multiple paragraphs") -> str:
        """Runs GraphRAG's own local/global/basic_search - the same retrieval route a real
        downstream query would use, including community reports - with `text` as the query, and
        returns its synthesized response as 'what the graph currently knows that's relevant to
        this'. `method` matters beyond retrieval quality: "local" and "basic" both do their own
        embedding-based similarity step internally, so they're unusable whenever the embedding
        endpoint is degraded/down (confirmed directly - the embedding endpoint timed out 3/3
        isolated tests while the completion endpoint answered instantly); "global" is pure
        completion-based map-reduce over community reports with no embedding dependency at all
        per graphrag's settings.yaml (no embedding_model_id under its config section), so it's
        also a real resilience option, not just an alternate retrieval style. Cached per (text,
        method, community_level, response_type): re-scoring the same doc across an Optuna study's
        trials that don't vary retrieval shouldn't re-pay a real query.

        Failures are cached too (as the exception itself, re-raised on every later lookup of the
        same key) - confirmed directly that a down embedding backend doesn't recover mid-retry (it
        just times out identically every time, not a transient blip), so without this an
        unattended multi-trial/multi-detector run would re-pay the full retry cost for the exact
        same document over and over. Only 1 attempt (no retry) for the same reason: retrying a
        call that's already confirmed to hang for the full timeout just multiplies the wait for no
        real chance of success - a later, independent document's call is effectively the next real
        "retry" anyway. Deliberately RAISES on failure rather than falling back to "" - a document
        this couldn't retrieve real context for must be excluded from this detector's results
        (via detect_batch's existing skip-and-log handling), not silently scored on no signal and
        recorded indistinguishably from a document that got a genuine graph-grounded check."""
        from scripts.helpers import retry_api_call
        # Callers pass a whole document as the query; graphrag stacks ~13k tokens of its own data on
        # top, overflowing the server's context and dropping the doc. Truncate instead - exclusions
        # correlate with length, so they bias results rather than just thinning them.
        text = text[:MAX_QUERY_CHARS]
        key = (text, method, community_level, response_type)
        if key not in self._context_cache:
            try:
                # lambda, not a bare coroutine - a coroutine object can only be awaited once, so a
                # retry needs to construct a fresh one on each attempt.
                self._context_cache[key] = retry_api_call(
                    lambda: _run_sync(
                        query(text, method=method, community_level=community_level, response_type=response_type),
                        # 900s not 300s: a global search over every community report takes ~250s per
                        # document even on a full H200, so 300s dropped slow docs as if broken.
                        timeout=900.0,
                    ),
                    retries=1,
                )
                self._persist_context(key, self._context_cache[key])
            except Exception as e:
                logging.warning(f"query_context: giving up on this query after persistent failure ({e}) - "
                                 f"this document will be excluded from results, not scored on empty context. "
                                 f"Caching the failure so later trials/detectors don't re-pay the retry cost.")
                self._context_cache[key] = e
        cached = self._context_cache[key]
        if isinstance(cached, Exception):
            raise cached
        # Collapse graphrag's canned refusal (and a blank response) to None, so callers can tell
        # "the graph has no evidence" apart from "the graph answered". See GRAPHRAG_NO_DATA_ANSWER
        # for why `if context` was the wrong guard. Returning None rather than "" makes the
        # distinction explicit at every call site instead of silently falsy.
        if not cached or not cached.strip() or cached.strip() == GRAPHRAG_NO_DATA_ANSWER:
            self._no_evidence_count += 1
            return None
        self._evidence_count += 1
        return cached

    def evidence_coverage(self) -> tuple[int, int]:
        """(calls that got real retrieved context, calls where the graph had nothing).

        Counted per query_context() call, cache hits included - so across an Optuna study the same
        document contributes once per trial that scores it, not once overall. Use the ratio, not
        the absolute counts, and read it as "share of retrieval attempts that found evidence".
        """
        return self._evidence_count, self._no_evidence_count

    def centrality(self, name: str) -> float:
        """Percentile-rank (0-1) of one entity's PageRank among all graph nodes - the 'how
        attractive a poisoning target is this' signal, since perturbing high-centrality nodes
        has outsized downstream effect (the intuition behind Nettack-style graph attacks).
        0.0 if `name` isn't a graph node."""
        if self._pagerank is None:
            # Double-checked under a lock, and _pagerank (the guard) is published LAST. Detectors
            # score concurrently in detect_batch's thread pool, so the previous unguarded version
            # let a second thread see _pagerank already assigned while _ranked_pageranks did not
            # exist yet, one line later - AttributeError, and only under whichever interleaving
            # happened to hit that window.
            with self._pagerank_lock:
                if self._pagerank is None:
                    pagerank = nx.pagerank(self._graph, weight="weight")
                    self._ranked_pageranks = sorted(pagerank.values())
                    self._pagerank = pagerank
        if name not in self._pagerank:
            return 0.0
        return bisect.bisect_right(self._ranked_pageranks, self._pagerank[name]) / len(self._ranked_pageranks)

    def hub_percentile(self, entity_names: list[str]) -> float:
        """Max centrality() over `entity_names` - convenience wrapper for callers (gnlic) that
        want a single per-document score rather than gtopo's per-pair use of centrality()."""
        return max((self.centrality(name) for name in entity_names), default=0.0)

    def has_edge(self, u: str, v: str) -> bool:
        return self._graph.has_edge(u, v)

    def is_known(self, name: str) -> bool:
        return name in self._graph

    def _community_map(self, level: int) -> dict[str, int]:
        """entity title -> GraphRAG's own Leiden community id at `level` (communities.parquet),
        not recomputed via networkx - lazily built and cached per level."""
        if level not in self._community_cache:
            comm_df = pd.read_parquet(gm.OUTPUT_DIR / "communities.parquet")
            ent_df = pd.read_parquet(gm.OUTPUT_DIR / "entities.parquet")
            id_to_title = dict(zip(ent_df["id"], ent_df["title"]))
            title_to_community: dict[str, int] = {}
            for _, row in comm_df[comm_df["level"] == level].iterrows():
                for eid in row["entity_ids"]:
                    title = id_to_title.get(eid)
                    if title:
                        title_to_community[title] = row["community"]
            self._community_cache[level] = title_to_community
        return self._community_cache[level]

    def _block_model(self, level: int) -> tuple[dict[str, int], dict[tuple, int], dict[int, int], int]:
        """(title -> community, {community pair: observed edges}, {community: summed degree},
        total edges placed in some community pair) at `level`.

        The raw counts behind _block_rates(), kept separately because pair_logodds() needs the
        observed/expected counts themselves - _block_rates() divides them away, and dividing
        early is what makes a never-observed pair indistinguishable from a rarely-observed one
        (see pair_logodds)."""
        if level not in self._block_model_cache:
            title_to_community = self._community_map(level)
            block_degree: dict[int, int] = {}
            for name in self._graph.nodes:
                c = title_to_community.get(name)
                if c is not None:
                    block_degree[c] = block_degree.get(c, 0) + self._graph.degree(name)
            block_edges: dict[tuple, int] = {}
            for u, v in self._graph.edges:
                cu, cv = title_to_community.get(u), title_to_community.get(v)
                if cu is None or cv is None:
                    continue
                key = (cu, cv) if cu <= cv else (cv, cu)
                block_edges[key] = block_edges.get(key, 0) + 1
            self._block_model_cache[level] = (
                title_to_community, block_edges, block_degree, sum(block_edges.values()),
            )
        return self._block_model_cache[level]

    def _block_rates(self, level: int) -> tuple[dict[str, int], dict[tuple, float]]:
        """Simple degree-corrected-ish block model estimated from the clean graph: for each
        pair of communities (a, b) at `level`, the rate of edges actually observed between them
        relative to their combined degree - a cheap stand-in for a fitted SBM's edge probability,
        used by block_surprise() to flag entity co-mentions that essentially never connect in the
        established graph."""
        if level not in self._block_cache:
            title_to_community, block_edges, block_degree, _ = self._block_model(level)
            block_rate = {
                key: count / (block_degree.get(key[0], 1) * block_degree.get(key[1], 1))
                for key, count in block_edges.items()
            }
            self._block_cache[level] = (title_to_community, block_rate)
        return self._block_cache[level]

    def pair_surprise(self, u: str, v: str, level: int = 2) -> float | None:
        """Percentile-rank (0-1) of how surprising a connection between u's and v's communities
        is, against the clean graph's own block rates - the per-pair primitive behind gtopo's
        'inc' signal. A type-preserving entity swap still reads fluently (no textual
        contradiction) but typically implies a connection between communities that essentially
        never connect in the established graph - invisible to text-only detectors by
        construction. Callers aggregate over a document's entity pairs themselves (e.g. a
        quantile, not a raw max - a single surprising pair souldn't be diluted, but a single
        saturated one shouldn't dominate either).

        Returns None - not 0.0 - when either entity has no community at this level, or the graph
        has no block rates to rank against. 0.0 means "least surprising connection in the graph",
        a claim there is no evidence for when the pair simply isn't placed; callers drop Nones
        instead of averaging a fabricated innocent value into the signal."""
        title_to_community, block_rate = self._block_rates(level)
        cu, cv = title_to_community.get(u), title_to_community.get(v)
        if cu is None or cv is None:
            return None
        key = (cu, cv) if cu <= cv else (cv, cu)
        surprise = -math.log(block_rate.get(key, 0.0) + 1e-9)
        all_surprises = self._surprise_cache.get(level)
        if all_surprises is None:
            # Sorted once per level, not per call: gtopo asks for up to max_pairs=400 pairs per
            # document across ~120 documents and ~27 trials, and this list is identical every time.
            all_surprises = sorted(-math.log(rate + 1e-9) for rate in block_rate.values())
            self._surprise_cache[level] = all_surprises
        n = len(all_surprises)
        if n == 0:
            return None
        return bisect.bisect_right(all_surprises, surprise) / n

    # --- gtopo2 primitives -------------------------------------------------------------
    # pair_surprise() above percentile-ranks its surprise against block_rate.values() - the rates
    # of community pairs that actually HAVE edges. A pair whose communities never connect hits the
    # +1e-9 floor, -log(1e-9) = 20.7, which outranks every observed rate and so ranks 1.0. Most
    # co-mentioned pairs in a real document are exactly that, so the measured result is
    # inc == 1.0 for 97% of wvc documents in BOTH classes (standalone AUC 0.501). The primitives
    # below are the unsaturated replacements: they keep magnitudes instead of ranks, and measure
    # local neighbourhood corroboration rather than community-level co-occurrence alone.

    LOGODDS_SMOOTHING = 0.5  # additive (Jeffreys-ish) prior on both observed and expected counts,
    # so a never-observed block pair yields a large-but-finite negative log-odds instead of -inf,
    # and rare-but-real pairs stay distinguishable from impossible ones.

    def pair_logodds(self, u: str, v: str, level: int = 2) -> float | None:
        """log(observed / expected) edges between u's and v's communities under a degree-corrected
        block model. Negative = the graph connects these two communities LESS than their degrees
        alone would predict, i.e. a link here is implausible; positive = they connect more.

        Unlike pair_surprise() this is a magnitude, not a percentile rank, so "never observed"
        and "observed a tenth as often as expected" stay distinct instead of both saturating at
        the top of the scale. Returns None (never 0.0, which would read as "exactly as expected")
        when either entity has no community at this level or the model is empty - callers drop
        Nones rather than averaging in a fabricated neutral value, matching pair_surprise()."""
        title_to_community, block_edges, block_degree, total = self._block_model(level)
        cu, cv = title_to_community.get(u), title_to_community.get(v)
        if cu is None or cv is None or not total:
            return None
        key = (cu, cv) if cu <= cv else (cv, cu)
        observed = block_edges.get(key, 0)
        # Standard degree-corrected null: expected edges between blocks a,b is k_a*k_b/(2m).
        expected = block_degree.get(cu, 0) * block_degree.get(cv, 0) / (2.0 * total)
        a = self.LOGODDS_SMOOTHING
        return math.log((observed + a) / (expected + a))

    def degree_of(self, name: str) -> int:
        """Graph degree of one entity, 0 if it isn't a node."""
        return self._graph.degree(name) if name in self._graph else 0

    def pair_support(self, u: str, v: str) -> tuple[float, int]:
        """(Adamic-Adar index, common-neighbour count) for u and v.

        The link-prediction view of "should these two entities plausibly be connected?" - a pair
        with many shared neighbours is corroborated by the surrounding graph even when no direct
        edge exists, while a pair with none is asserted out of nowhere. Adamic-Adar discounts
        shared neighbours that are hubs (1/log deg), so co-occurring through "UNITED STATES"
        counts for far less than through a specific club or season. Replaces gtopo v1's binary
        has_edge(), which was false for ~100% of sampled pairs in both classes and so carried no
        information. (0.0, 0) for unknown entities - a real absence of support, not an abstention.
        """
        if u not in self._graph or v not in self._graph:
            return 0.0, 0
        common = self._graph[u].keys() & self._graph[v].keys()
        aa = sum(1.0 / math.log(d) for d in (self._graph.degree(w) for w in common) if d > 1)
        return aa, len(common)

    def hop_distance(self, u: str, v: str, cap: int = 4) -> int:
        """Shortest-path distance between u and v, capped at `cap` (returns cap+1 for anything
        further, unreachable, or not in the graph). Capping is what keeps this affordable: a
        bidirectional BFS that may stop at depth `cap` costs a small neighbourhood walk instead of
        a full traversal, and distances beyond ~4 hops are not meaningfully different for this
        purpose anyway."""
        if u not in self._graph or v not in self._graph:
            return cap + 1
        if u == v:
            return 0
        # BFS truncated at `cap` - this is what actually makes the cap a cost saving rather than
        # just a clamp on the answer. networkx has no bidirectional_shortest_path_length (only
        # bidirectional_shortest_path, which returns a path and takes no depth limit), so the
        # cutoff form of single_source is the one that stops early. Nodes beyond the cutoff are
        # simply absent from the result, which is the same answer as unreachable.
        lengths = nx.single_source_shortest_path_length(self._graph, u, cutoff=cap)
        return lengths.get(v, cap + 1)

    def egonet_stats(self, entity_names: list[str]) -> dict[str, float] | None:
        """Structure of the subgraph the graph itself induces on `entity_names` - the OddBall
        (Akoglu et al.) reading of "does this set of entities hang together the way a real
        document's entities do?". A document asserting relations the graph doesn't share yields a
        sparse, fragmented induced subgraph.

        Returns n_nodes/n_edges (the two quantities the Egonet Density Power Law E ~ N^alpha
        relates - the caller fits alpha on clean documents and scores the residual), plus the
        fragmentation shape. None for fewer than 2 known entities, where none of it is defined."""
        known = [n for n in entity_names if n in self._graph]
        n = len(known)
        if n < 2:
            return None
        sub = self._graph.subgraph(known)
        components = list(nx.connected_components(sub))
        return {
            "n_nodes": float(n),
            "n_edges": float(sub.number_of_edges()),
            "isolated_frac": sum(1 for name in known if sub.degree(name) == 0) / n,
            "component_frac": len(components) / n,
            "lcc_frac": max(len(c) for c in components) / n,
        }

    def find_entities_in_text(self, text: str) -> list[str]:
        """Entities (by title) from the graph that are mentioned in `text`, matched on
        word boundaries (not raw substring) so a short title like 'ER' doesn't match
        inside an unrelated word like 'American'. The cheap substring check runs first
        as a pre-filter; the regex boundary check only runs on what survives it.
        """
        text_lower = text.lower()
        matches = []
        for name, name_lower in self._lowered_titles:
            if name_lower not in text_lower:
                continue
            if re.search(rf"\b{re.escape(name_lower)}\b", text_lower):
                matches.append(name)
        return matches

    def related_texts(self, entity_names: list[str], hops: int = 1, top_n: int = 5) -> list[str]:
        """Raw source text units connected to `entity_names`: each seed entity's own
        top `top_n` text units, plus its top `top_n` graph-neighbours (ranked by edge
        weight) each contributing their own top `top_n` text units, repeated for
        `hops` steps. Capped this way so a hub entity (e.g. degree 391 for
        "UNITED STATES") doesn't flood the pool the way taking every neighbour's every
        text unit did - this is what made note_max saturate near 1.0 on clean docs too.
        """
        seen_entities = set(entity_names)
        seen_text_ids = set()
        texts = []

        def add_texts(node: str, limit: int) -> None:
            for text_unit_id in list(self._graph.nodes[node].get("text_unit_ids", []))[:limit]:
                if text_unit_id in seen_text_ids:
                    continue
                seen_text_ids.add(text_unit_id)
                text = self._id_to_text.get(text_unit_id)
                if text:
                    texts.append(text)

        for name in entity_names:
            if name in self._graph:
                add_texts(name, top_n)

        frontier = set(entity_names)
        for _ in range(hops):
            next_frontier = set()
            for name in frontier:
                if name not in self._graph:
                    continue
                neighbors = sorted(
                    self._graph.neighbors(name),
                    key=lambda nb: self._graph[name][nb].get("weight", 0.0),
                    reverse=True,
                )[:top_n]
                for neighbor in neighbors:
                    if neighbor in seen_entities:
                        continue
                    seen_entities.add(neighbor)
                    next_frontier.add(neighbor)
                    add_texts(neighbor, top_n)
            frontier = next_frontier
        return texts

    def related_texts_weighted(self, entity_names: list[str], hops: int = 1, top_n: int = 5) -> list[tuple[str, float]]:
        """Like related_texts(), but pairs each text with a relevance weight instead of
        treating every note equally: a neighbour's weight is the product of edge weights
        along the traversal path from whichever seed entity led to it (so a note reached
        through two weak edges counts for less than one reached through a single strong
        edge, and notes further out decay relative to closer ones). Seed entities'
        own texts - found directly in the document, not via any edge - get the max
        weight seen among the hop-neighbours (or 1.0 if there are none), since a direct
        mention is at least as relevant as anything reached from it.
        """
        seen_entities = set(entity_names)
        seen_text_ids = set()
        texts: list[tuple[str, float]] = []
        seed_slots: list[int] = []

        def add_texts(node: str, limit: int, weight: float, seed: bool) -> None:
            for text_unit_id in list(self._graph.nodes[node].get("text_unit_ids", []))[:limit]:
                if text_unit_id in seen_text_ids:
                    continue
                seen_text_ids.add(text_unit_id)
                text = self._id_to_text.get(text_unit_id)
                if text:
                    if seed:
                        seed_slots.append(len(texts))
                    texts.append((text, weight))

        for name in entity_names:
            if name in self._graph:
                add_texts(name, top_n, weight=0.0, seed=True)  # placeholder, backfilled below

        frontier = {name: 1.0 for name in entity_names if name in self._graph}
        for _ in range(hops):
            next_frontier: dict[str, float] = {}
            for name, path_weight in frontier.items():
                neighbors = sorted(
                    self._graph.neighbors(name),
                    key=lambda nb: self._graph[name][nb].get("weight", 0.0),
                    reverse=True,
                )[:top_n]
                for neighbor in neighbors:
                    if neighbor in seen_entities:
                        continue
                    seen_entities.add(neighbor)
                    neighbor_weight = path_weight * self._graph[name][neighbor].get("weight", 0.0)
                    next_frontier[neighbor] = neighbor_weight
                    add_texts(neighbor, top_n, weight=neighbor_weight, seed=False)
            frontier = next_frontier

        max_weight = max((w for _, w in texts if w > 0.0), default=1.0)
        for i in seed_slots:
            text, _ = texts[i]
            texts[i] = (text, max_weight)
        return texts
