"""
GraphRAG manager: index checking, indexing, and querying via the graphrag Python API.
All public functions are async. Call them with asyncio.run() or await from an async context.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from graphrag.api import build_index, basic_search, global_search, local_search
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config as _load_graphrag_config
from graphrag.logger.progress import Progress
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from tqdm import tqdm
import networkx as nx

from scripts.helpers import log_time

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRAPHRAG_ROOT = PROJECT_ROOT / "graphrag-api/graphrag"

OUTPUT_DIR = GRAPHRAG_ROOT / "output"

# Same bar style as helpers.progress_bar(), so indexing progress looks and
# behaves like every other progress bar in this project (ETA included).
_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


class _TqdmWorkflowCallbacks:
    """WorkflowCallbacks that render indexing progress as a single tqdm bar
    per workflow, matching the project's other progress bars (ETA included)
    instead of graphrag's own crude percentage-fill callback.
    """

    def __init__(self) -> None:
        self._bar: tqdm | None = None

    def pipeline_start(self, names: list[str]) -> None:
        tqdm.write(f"Starting pipeline with workflows: {', '.join(names)}")

    def pipeline_end(self, results: list[PipelineRunResult]) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        tqdm.write("Pipeline complete")

    def workflow_start(self, name: str, instance: object) -> None:
        if self._bar is not None:
            self._bar.close()
        self._bar = tqdm(total=1, desc=name, ncols=80, bar_format=_BAR_FORMAT)

    def workflow_end(self, name: str, instance: object) -> None:
        if self._bar is not None:
            self._bar.n = self._bar.total
            self._bar.refresh()
            self._bar.close()
            self._bar = None

    def progress(self, progress: Progress) -> None:
        if self._bar is None or not progress.total_items:
            return
        if self._bar.total != progress.total_items:
            self._bar.total = progress.total_items
        self._bar.n = progress.completed_items or 0
        self._bar.refresh()

    def pipeline_error(self, error: BaseException) -> None:
        tqdm.write(f"Pipeline error: {error}")


def set_active_corpus(corpus_name: str) -> None:
    """Switch GraphRAG's input and output directories to a named corpus subdirectory.

    Updates settings.yaml in-place so all subsequent graphrag calls
    (indexing, querying, vector store) target that corpus.  Also updates the
    module-level OUTPUT_DIR so check_index_status / _load_dataframes stay in sync.

    Args:
        corpus_name: Subdirectory name under input/ and output/,
                     e.g. "wvc" → reads from input/wvc/, writes to output/wvc/.
    """
    global OUTPUT_DIR
    import yaml

    settings_path = GRAPHRAG_ROOT / "settings.yaml"
    cfg = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    # graphrag 2.2.1 reads "input"/"output"; "input_storage"/"output_storage" is 3.x naming and is
    # silently IGNORED (extra keys don't error). Writing only the *_storage pair once cost two hours
    # of indexing the wrong corpus. Both spellings written; the 2.2.1 pair is authoritative.
    for key, value in (("input", f"input/{corpus_name}"), ("output", f"output/{corpus_name}")):
        cfg.setdefault(key, {})["base_dir"] = value
        cfg.setdefault(f"{key}_storage", {})["base_dir"] = value
    # vector_store holds NAMED store configs, each with its own nested db_uri - a flat
    # vector_store.db_uri is a stray key that graphrag's schema rejects at the next real index run.
    vector_store = cfg.setdefault("vector_store", {})
    vector_store.pop("db_uri", None)  # strip a stray flat key left by the old buggy version
    for store_cfg in vector_store.values():
        if isinstance(store_cfg, dict):
            store_cfg["db_uri"] = f"output/{corpus_name}/lancedb"

    settings_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    OUTPUT_DIR = GRAPHRAG_ROOT / "output" / corpus_name
    logger.info("Active GraphRAG corpus set to '%s' (output: %s)", corpus_name, OUTPUT_DIR)


def sync_corpus_to_graphrag_input(corpus_name: str, corpus_jsonl: Path, exclude_indices: set[int] | None = None) -> None:
    """Convert a dataset's corpus.jsonl into the JSON array GraphRAG's json loader expects.

    GraphRAG's json input loader expects a top-level array ([{...}, ...]), not
    newline-delimited JSON, so corpus.jsonl can't just be copied in. Writes the
    result to input/<corpus_name>/corpus.json under GRAPHRAG_ROOT and removes any
    stale .jsonl files left over from a previous sync. `exclude_indices`, if given, skips
    those positional line indices - used to build a benchmark base index that has never
    seen the documents held out for testing.
    """
    graphrag_input = GRAPHRAG_ROOT / "input" / corpus_name
    graphrag_input.mkdir(parents=True, exist_ok=True)

    # Stream the JSONL file to avoid loading entire corpus into memory
    records = []
    with open(corpus_jsonl, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            if exclude_indices and i in exclude_indices:
                continue
            line = line.strip()
            if line:
                records.append(json.loads(line))

    (graphrag_input / "corpus.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    for stale in graphrag_input.glob("*.jsonl"):
        stale.unlink()
    logger.info("Synced %s (%d docs) into GraphRAG input for '%s'.", corpus_jsonl, len(records), corpus_name)


def ensure_dataset_synced(dataset: str, conf: dict) -> None:
    """Copies a dataset into grag-data/input if missing, then syncs it into GraphRAG's
    input dir as a JSON array. Both graphrag_test() and the benchmark need this done
    before indexing."""
    project_root = str(PROJECT_ROOT)
    input_path = os.path.join(project_root, "grag-data", "input", dataset, "corpus.jsonl")
    if not os.path.isfile(input_path):
        logger.debug("Copying dataset %s to GraphRAG input folder...", dataset)
        data_root = conf["attack_config"]["data_root"].replace("/", os.sep)
        src_path = os.path.join(project_root, data_root, dataset)
        if not os.path.isdir(src_path):
            # Datasets outside the logicPoison_original_Dataset submodule (e.g. graphrag_under_fire,
            # built by setup_dataset.py) live directly under datasets/<dataset>/ instead.
            fallback_src = os.path.join(project_root, "datasets", dataset)
            if os.path.isdir(fallback_src):
                src_path = fallback_src
        dst_path = os.path.join(project_root, "grag-data", "input", dataset)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    if dataset != "wvc":
        sync_corpus_to_graphrag_input(dataset, Path(input_path))


_bg_loop: "asyncio.AbstractEventLoop | None" = None
_bg_thread: threading.Thread | None = None
_bg_loop_lock = threading.Lock()


def _get_background_loop() -> "asyncio.AbstractEventLoop":
    """One event loop, running forever on one daemon thread, reused for every _run_sync() call
    for the rest of the process's life - lazily started on first use. Not "a fresh loop per
    call": that pattern (tried first) broke litellm's internal LoggingWorker, which lazily binds
    a persistent asyncio.Queue to whichever event loop is active the first time it's touched -
    every later call on a NEW event loop then hit "Queue... is bound to a different event loop"
    and hung the whole process (confirmed - this is what actually happened). A single long-lived
    loop keeps every async library's internal state consistently bound to the same loop."""
    global _bg_loop, _bg_thread
    with _bg_loop_lock:
        if _bg_loop is None:
            _bg_loop = asyncio.new_event_loop()
            _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
            _bg_thread.start()
        return _bg_loop


def _run_sync(coro, timeout: float | None = None):
    """Runs an async coroutine to completion from sync code, even when called from inside an
    already-running event loop (detect_batch() is sync but needs to trigger async indexing) -
    submits it to a single persistent background event loop (see _get_background_loop) rather
    than spinning up a new thread+loop per call.

    `timeout` (seconds), if given, bounds how long we *wait* and requests cancellation of the
    coroutine if it elapses - not a guarantee the coroutine stops immediately (asyncio
    cancellation is cooperative), but the calling thread regains control either way. Found the
    hard way: query_context()'s local_search occasionally hangs inside GraphRAG's own internal
    LLM client with no exception ever raised (not a timeout/connection error our retry_api_call
    could catch - the call just never returns), which previously meant an unbounded wait froze
    the entire benchmark process indefinitely (confirmed via two separate hangs showing ~0 CPU
    and zero open sockets for 5-20+ minutes straight). None (the default, used by the real
    indexing path) waits forever, since a genuine multi-hour reindex is expected to legitimately
    take a long time."""
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        # Cancellation is cooperative and this task already proved it never yields, so it can
        # go on sitting on `loop` forever - every later call queuing behind the same stuck
        # resource on the same loop would then time out too, turning one hang into a permanent
        # one. Retiring the loop reference (the old thread/loop keep running independently;
        # nothing forcibly stops them) makes the next call use a fresh loop instead.
        #
        # That alone isn't enough: graphrag.language_model.manager.ModelManager caches chat/
        # embedding model instances process-wide by name, and each one's fnllm rate limiter
        # binds its internal asyncio.Semaphore/Future to whichever loop first used it - reusing
        # a cached model on the NEW loop crashes instantly with "attached to a different loop"
        # (confirmed: job 46065449, 2350 such errors in 11 minutes after a loop-only fix).
        # Clearing the cache here forces fresh, new-loop-bound model instances on the next call;
        # a task still in flight on the old loop keeps its own already-held reference, unaffected.
        global _bg_loop
        with _bg_loop_lock:
            if _bg_loop is loop:
                _bg_loop = None
        try:
            from graphrag.language_model.manager import ModelManager
            manager = ModelManager()
            for name in list(manager.list_chat_models()):
                manager.remove_chat(name)
            for name in list(manager.list_embedding_models()):
                manager.remove_embedding(name)
        except Exception:
            logger.exception("Failed to clear ModelManager cache after a _run_sync timeout.")
        raise TimeoutError(f"_run_sync: coroutine did not complete within {timeout}s - abandoning (likely a hung internal GraphRAG call).")


def _ingest_document(corpus: str, text: str) -> bool:
    """Appends `text` as a new document to the active corpus's GraphRAG input, then runs an
    incremental update reindex (not a full rebuild) so it's genuinely merged into the graph.
    Low-level file/reindex primitive - GraphConnector.ingest_document() wraps this and also
    refreshes its own in-memory graph state afterward."""
    input_path = GRAPHRAG_ROOT / "input" / corpus / "corpus.json"
    records = json.loads(input_path.read_text(encoding="utf-8"))
    new_id = f"ingest_{uuid.uuid4().hex[:12]}"
    records.append({"_id": new_id, "title": new_id, "text": text})
    input_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return _run_sync(run_indexing(is_update_run=True))


API_DIR = PROJECT_ROOT / "graphrag-api"
API_PORT = 8000

# Cap on the document text used as a *query* in query_context() - see the rationale there.
MAX_QUERY_CHARS = 8000

# graphrag's canned refusal (NO_DATA_ANSWER) when global search scores every community report at 0.
# A SOFT failure: the call succeeds and the string is non-empty, so `if context` treats it as real
# context - which fed detectors an apology as evidence. Global search only; local/basic never emit it.
GRAPHRAG_NO_DATA_ANSWER = "I am sorry but I am unable to answer this question given the provided data."


class NoGraphEvidence(RuntimeError):
    """Raised by a grag-aware detector when the graph returned no usable evidence for a document.

    Distinct from an API failure: the retrieval call succeeded, the graph simply had nothing
    relevant. Detectors raise this instead of scoring on no signal, so detect_batch can exclude
    the document and report coverage separately - "the detector had no evidence for X% of
    documents" is a finding about GraphRAG-based auditing, not a score of 'innocent'.
    """

# Disk-backed store for query_context() results, one JSONL per corpus - see
# GraphConnector._load_persisted_contexts. Appends come from detect_batch's worker threads.
CONTEXT_CACHE_DIR = PROJECT_ROOT / "results" / "context_cache"
_context_persist_lock = threading.Lock()


def restart_graphrag_api() -> None:
    """Kill any process on API_PORT and relaunch graphrag-api/api.py in the background.

    api.py loads its parquet/lancedb state once at startup, so it must be
    restarted after reindexing for queries to see fresh data.
    """
    try:
        pids = subprocess.run(
            ["lsof", "-ti", f":{API_PORT}"], capture_output=True, text=True
        ).stdout.split()
        for pid in pids:
            subprocess.run(["kill", pid])
        if pids:
            logger.info("Stopped existing GraphRAG API process(es) on port %d: %s", API_PORT, pids)
    except FileNotFoundError:
        logger.warning("lsof not available — cannot check for existing GraphRAG API process.")

    subprocess.Popen(
        [sys.executable, "api.py"],
        cwd=API_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info("Relaunched GraphRAG API server on port %d.", API_PORT)

# Parquet files produced by a completed index run.
# covariates.parquet is optional (only when extract_claims is enabled).
REQUIRED_PARQUETS = [
    "entities.parquet",
    "communities.parquet",
    "community_reports.parquet",
    "text_units.parquet",
    "relationships.parquet",
    "documents.parquet",
]

def load_graph() -> nx.Graph:
    """Build a NetworkX graph from the indexed entities and relationships."""
    dfs = _load_dataframes()
    G = nx.Graph()

    for _, row in dfs["entities"].iterrows():
        G.add_node(
            row["title"],
            type=row.get("type", ""),
            description=row.get("description", ""),
            text_unit_ids=row.get("text_unit_ids", []),
        )

    for _, row in dfs["relationships"].iterrows():
        G.add_edge(row["source"], row["target"], weight=row.get("weight", 1.0), description=row.get("description", ""))

    return G

def check_index_status() -> dict:
    """Return a dict describing whether the index is complete.

    Keys:
        output_dir_exists (bool)
        missing_files     (list[str])
        is_complete       (bool)
    """
    missing = [f for f in REQUIRED_PARQUETS if not (OUTPUT_DIR / f).exists()]
    lancedb_ok = any((OUTPUT_DIR / "lancedb").glob("*.lance"))
    return {
        "output_dir_exists": OUTPUT_DIR.exists(),
        "missing_files": missing,
        "is_complete": len(missing) == 0 and lancedb_ok,
    }


def _config():
    return _load_graphrag_config(GRAPHRAG_ROOT)


@contextmanager
def _graphrag_cwd():
    """graphrag's config loader chdirs into GRAPHRAG_ROOT to resolve the relative
    storage/prompt paths in settings.yaml and leaves it there; the indexing
    workflows and search calls that follow rely on that CWD staying put while
    they run. Restore the caller's original CWD only once they're done, so the
    rest of this project (which resolves paths against the project root) isn't
    affected.
    """
    original_cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _load_dataframes() -> dict:
    """Load all parquet files from the output directory into DataFrames."""
    dfs: dict = {}
    for fname in REQUIRED_PARQUETS:
        path = OUTPUT_DIR / fname
        dfs[fname.replace(".parquet", "")] = pd.read_parquet(path)
    # covariates are optional (claims pipeline)
    cov_path = OUTPUT_DIR / "covariates.parquet"
    dfs["covariates"] = pd.read_parquet(cov_path) if cov_path.exists() else None
    return dfs


async def run_indexing(method: IndexingMethod = IndexingMethod.Standard, is_update_run: bool = False) -> bool:
    """Run the GraphRAG indexing pipeline. `is_update_run=True` runs GraphRAG's incremental
    update workflows instead of a full reindex - only new documents (vs. the previous run's
    output) get processed and merged into the existing graph."""
    with _graphrag_cwd():
        config = _config()
        results = await build_index(
            config=config,
            method=method,
            is_update_run=is_update_run,
            callbacks=[_TqdmWorkflowCallbacks()],
        )
        for r in results:
            status = "ERROR" if r.errors else "done"
            logger.info("[graphrag] %-40s %s", r.workflow, status)
        errors = [r for r in results if r.errors]
        for r in errors:
            logger.error("Indexing error in workflow '%s': %s", r.workflow, r.errors)
        return len(errors) == 0


@log_time("graphrag_init")
async def graphrag_init() -> None:
    """Indexes the active corpus if it isn't already complete, then restarts the API
    server so it picks up the fresh index."""
    status = check_index_status()
    if status["is_complete"]:
        logger.debug("GraphRAG index is complete.")
        return
    logger.info("GraphRAG index is incomplete. Starting indexing pipeline...")
    if not await run_indexing():
        logger.error("Indexing failed. Check app.log for details.")
        return
    restart_graphrag_api()
    logger.info("Indexing completed, GraphRAG API server restarted with fresh index.")


async def query(
    query_text: str,
    method: str = "local",
    community_level: int = 2,
    response_type: str = "multiple paragraphs",
) -> str:
    """Query the GraphRAG index.

    Parameters
    ----------
    query_text:
        The natural-language question.
    method:
        "local"  — entity-centric, good for specific questions.
        "global" — community-level, good for broad/thematic questions.
        "basic"  — embedding-only, fastest but shallowest.
    community_level:
        Graph community depth to use (higher = broader context).
    response_type:
        Natural-language instruction passed to the LLM for output format,
        e.g. "multiple paragraphs", "single sentence", "bullet list".

    Returns
    -------
    The LLM response as a string.
    """
    with _graphrag_cwd():
        config = _config()
        dfs = _load_dataframes()

        if method == "local":
            response, _ = await local_search(
                config=config,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                text_units=dfs["text_units"],
                relationships=dfs["relationships"],
                covariates=dfs["covariates"],
                community_level=community_level,
                response_type=response_type,
                query=query_text,
            )
        elif method == "global":
            response, _ = await global_search(
                config=config,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                community_level=community_level,
                # Rate communities top-down and map over only the relevant ones, rather than every
                # report at community_level. Measured on wvc: 335 reports (~277k tokens at level<=2)
                # split into ~24 map calls per document, ~24k generated tokens, ~250s per document
                # even on a full H200 - the dominant cost in a gnlic/grag_as_a_judge study by far.
                # Needs dynamic_search_use_summary=True in settings.yaml (the slurm scripts set it):
                # it defaults to rating against full_content, which would cost about as much as the
                # map calls it saves.
                dynamic_community_selection=True,
                response_type=response_type,
                query=query_text,
            )
        elif method == "basic":
            response, _ = await basic_search(
                config=config,
                text_units=dfs["text_units"],
                response_type=response_type,
                query=query_text,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'local', 'global', or 'basic'.")

    return response


