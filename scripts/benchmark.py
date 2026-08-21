"""Runs the detector benchmark: Optuna tuning, insertion/edit detection, and chart output."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

from scripts.helpers import printv, log_time, make_progress_ticker
from scripts.graphrag_manager import (
    check_index_status, restart_graphrag_api, set_active_corpus,
    sync_corpus_to_graphrag_input, ensure_dataset_synced, graphrag_init, run_indexing,
)
from scripts.connectors import EmbeddingConnector, GraphConnector
import scripts.version as v
from scripts.version import VersionStoreConfig


def _optimize_detector_params(conf: dict, dataset: str, wanted_names: set, all_poisoned_diffs: list, all_clean_diffs: list,
                               reserved: int, embedding_connector, graph_connector, nli_scorer, use_dva: bool,
                               val_size: int | None = None) -> dict:
    """Runs Optuna tuning (scripts/optimize.py) for whichever of `wanted_names` are tunable, on a
    validation slice starting after the `reserved` diffs the real benchmark reports on, against
    `use_dva`'s mode. Returns {} if disabled or nothing to tune. Runs against whatever corpus state
    is currently checked out - unlike the final report loop, tuning does not ingest/roll back per
    trial (would multiply the already-expensive grag-aware reindex cost by n_trials); a
    hyperparameter search only needs a reasonable proxy signal, not the same leakage guarantee as
    the reported numbers.

    Structural params (hops/k_neighbors/variant/...) can genuinely differ between what's best for
    n-mode vs dva-mode (llm_as_a_judge/grag_as_a_judge's `variant` especially - n and dva draw from
    disjoint prompts.jsonl pools), so each mode gets its own best_params file
    (`{dataset}_n.json`/`{dataset}_dva.json`), not a shared one - callers needing both modes build
    two separate sets of detector instances, one per file."""
    opt_conf = conf["benchmark"].get("optimize_params", {})
    if not opt_conf.get("enabled", False):
        return {}

    from scripts.optimize import optimize_all, tunable_detector_names

    if val_size is None:
        val_size = opt_conf.get("val_size", 10)
    poisoned_val = all_poisoned_diffs[reserved:reserved + val_size]
    clean_val = all_clean_diffs[reserved:reserved + val_size]
    tunable_names = [n for n in tunable_detector_names() if n in wanted_names]

    if not (tunable_names and poisoned_val and clean_val):
        printv("optimize_params enabled but nothing to tune or empty validation split - skipping.", level="warning")
        return {}

    mode_suffix = "dva" if use_dva else "n"
    best_params_dir = Path(opt_conf.get("best_params_dir", "results/detector_best_params/"))
    return optimize_all(
        tunable_names, clean_val, poisoned_val,
        n_trials=opt_conf.get("n_trials", 20),
        best_params_path=best_params_dir / f"{dataset}_{mode_suffix}.json",
        history_dir=best_params_dir / f"{dataset}_{mode_suffix}_trials",
        force=opt_conf.get("force_reoptimize", False),
        embedding_connector=embedding_connector, graph_connector=graph_connector, nli_scorer=nli_scorer,
        use_dva=use_dva,
    )


def _load_prompt_variants(path: str | Path) -> dict[str, int]:
    """Loads the cross-dataset-selected prompt variant per judge detector (see
    scripts/select_prompt_variant.py) - deliberately separate from best_params/Optuna, since which
    prompt to use is a single fixed deployment choice, not something to tune per dataset/attack
    type (a real system doesn't know in advance which attack it's facing). Falls back to variant 0
    (prompts.jsonl's first-listed variant per detector) if selection hasn't been run yet."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {name: entry["variant"] for name, entry in data.items()}


def _build_detector_objects(conf: dict, embedding_connector, graph_connector, nli_scorer, best_params: dict) -> list:
    """Instantiates every known detector, merging in Optuna's tuned params where available."""
    from scripts.detector import (
        rand, rand1, llm_as_a_judge, grag_as_a_judge, perplexity_filtering,
        embedding_outlier, nli_contradiction,
        gnlic, gtopo, gtopo2,
    )
    from scripts.optimize import search_space
    optimize_threshold = conf["benchmark"].get("optimize_threshold", False)
    prompt_variants = _load_prompt_variants(
        conf["benchmark"].get("prompt_variants_path", "results/detector_best_params/prompt_variants.json")
    )

    # Cached thresholds, keyed by class name, applied after construction below. results() writes
    # them as threshold/threshold_edit (the attributes are threshold_n/threshold_dva), but every
    # detector names its *constructor* arg differently - abs_threshold, z_threshold,
    # direct_threshold, or none at all for the judges - so they can't travel as kwargs.
    restored: dict[str, dict] = {}

    def tuned(name: str, **defaults) -> dict:
        # Structural params always come from Optuna; thresholds only when optimize_threshold is
        # off - otherwise finalize() immediately re-derives them from this run's own records anyway.
        found = best_params.get(name, {})
        # Only params the CURRENT search space still contains. best_params files are reused across
        # runs, so without this filter a param dropped from the grid keeps being reinstated from
        # cache - graphrag_under_fire_*_n.json still carries method="global", the setting removed
        # for retrieving only ~21/120 documents, and it would silently survive every future run.
        space = search_space(name)
        cached = {k: v for k, v in found.get("params", {}).items() if k in space} if space else {}
        kwargs = {**defaults, **cached, "optimize_threshold": optimize_threshold}
        if not optimize_threshold:
            restored[name] = {attr: found[key] for key, attr in
                              (("threshold", "threshold_n"), ("threshold_edit", "threshold_dva"))
                              if key in found}
        return kwargs

    objects = [
        # Detectors that don't need the (currently degraded/down) embedding endpoint first - see
        # scripts/optimize.py's _TUNABLE ordering comment for why.
        rand(optimize_threshold=optimize_threshold),
        rand1(optimize_threshold=optimize_threshold),
        # variant: Optuna-tuned per-dataset if optimize_params ran for this detector, else the
        # older cross-dataset select_prompt_variant.py choice, else prompts.jsonl's variant 0.
        llm_as_a_judge(**tuned("llm_as_a_judge", variant=prompt_variants.get("llm_as_a_judge", 0))),
        perplexity_filtering(optimize_threshold=optimize_threshold),
        # embedding_connector is the no-evidence fallback path for every grag-aware NLI detector:
        # when the graph pulls no notes they score against embedding neighbours instead of
        # emitting the 0.0 that a fully-corroborated document would get.
        gnlic(graph_connector, nli_scorer, embedding_connector=embedding_connector, **tuned("gnlic")),
        gtopo(graph_connector, **tuned("gtopo")),
        # v1 stays in the list alongside it - gtopo2 is a separate detector, not a replacement, so
        # the existing v1 records stay comparable (see the class docstring).
        gtopo2(graph_connector, **tuned("gtopo2")),
        grag_as_a_judge(graph_connector, **tuned("grag_as_a_judge", variant=prompt_variants.get("grag_as_a_judge", 0))),
        embedding_outlier(embedding_connector, **tuned("embedding_outlier")),
        nli_contradiction(embedding_connector, nli_scorer, **tuned("nli_contradiction", k_neighbors=3)),
    ]  # TODO: add other detectors, when implemented
    for obj in objects:
        for attr, value in restored.get(type(obj).__name__, {}).items():
            setattr(obj, attr, value)
    return objects


def _find_tag(label: str) -> str | None:
    """Version-store tags are `f"{timestamp}_{label}"` (see scripts/version.py's save()), so a
    tag search is a suffix match on the label, not an exact match on the commit message."""
    return next((ver["tag"] for ver in v.list_versions() if ver["tag"].endswith(label)), None)


def _force_rmtree(path: Path) -> None:
    """shutil.rmtree, but clears read-only bits first - git marks its object files read-only,
    which makes a plain rmtree fail with PermissionError on Windows once a version-store repo
    has been initialized inside `path`."""
    def _on_error(func, target, _exc_info):
        os.chmod(target, stat.S_IWRITE)
        func(target)
    shutil.rmtree(path, onerror=_on_error)


async def _ensure_base_versions(dataset: str, conf: dict, clean_corpus_path: str, test_record_ids: set[int],
                                 negative_diffs: list[dict], build_with_negatives: bool,
                                 exclude_test_docs: bool = True) -> tuple[str, str | None]:
    """Builds (or reuses, if already tagged) a two-tier version-store snapshot for `dataset`:

    - `benchmark_base_{dataset}`: if exclude_test_docs, every test document (report split *and*
      optimization split, positive *and* negative) is excluded from the very first index build -
      not just "whatever's already indexed", which would still contain every test doc's original.
      If not exclude_test_docs (wvc/LogicPoison-only datasets: negatives come from a genuinely
      separate source - see run_benchmark's `dataset == "wvc"` branch - so there's nothing to leak
      and the whole corpus is used as-is), this just tags the already-complete full index.
    - `benchmark_base_with_negatives_{dataset}`: the tier above, with every negative (clean)
      test doc genuinely ingested on top - built once and reused by every detector's run, since
      ingestion is a corpus-wide disk mutation, identical regardless of which detector triggers it.

    Both are cached: a prior run's tags are reused as-is rather than rebuilt.

    The version store gets its own dedicated git+dvc repo rooted at the output folder itself
    (not the project's real repo): `graphrag-api` is a submodule whose own .gitignore already
    excludes `graphrag/output/`, so a nested repo there is invisible to both the outer project's
    git and the submodule's - v.restore()'s `git checkout <tag>` would otherwise detach HEAD on
    whichever real repo it ran against, and litter its history with "benchmark_base_*" commits.
    """
    output_dir = Path("graphrag-api/graphrag/output") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    v.init(VersionStoreConfig(repo_root=str(output_dir), tracked_folders=[output_dir]))

    base_label = f"benchmark_base_{dataset}"
    base_tag = _find_tag(base_label)
    if base_tag:
        v.restore(base_tag)
    elif not exclude_test_docs:
        printv(f"No test-doc exclusion needed for {dataset} - tagging the existing full index as-is.", level="info")
        set_active_corpus(dataset)
        if not check_index_status()["is_complete"]:
            sync_corpus_to_graphrag_input(dataset, Path(clean_corpus_path))
            if not await run_indexing():
                raise RuntimeError(f"Failed to build benchmark base index for {dataset}")
            restart_graphrag_api()
        base_tag = v.save(label=base_label)
    else:
        printv(f"Building test-doc-excluded base index for {dataset}...", level="info")
        sync_corpus_to_graphrag_input(dataset, Path(clean_corpus_path), exclude_indices=test_record_ids)
        set_active_corpus(dataset)
        if output_dir.exists():
            _force_rmtree(output_dir)  # force a real full reindex, not "already complete, skip" -
            # also wipes the .git just initialized above (read-only object files need _force_rmtree
            # on Windows), so re-init after rebuilding below.
        if not await run_indexing():
            raise RuntimeError(f"Failed to build benchmark base index for {dataset}")
        restart_graphrag_api()
        output_dir.mkdir(parents=True, exist_ok=True)
        v.init(VersionStoreConfig(repo_root=str(output_dir), tracked_folders=[output_dir]))
        base_tag = v.save(label=base_label)

    if not build_with_negatives:
        return base_tag, None

    with_negatives_label = f"benchmark_base_with_negatives_{dataset}"
    with_negatives_tag = _find_tag(with_negatives_label)
    if not with_negatives_tag:
        printv(f"Ingesting {len(negative_diffs)} negative docs into base for {dataset}...", level="info")
        graph_connector = GraphConnector(corpus=dataset)
        for diff in negative_diffs:
            graph_connector.ingest_document(diff["new_text"])
        with_negatives_tag = v.save(label=with_negatives_label)
    return base_tag, with_negatives_tag


def _make_ingest_fn(d, ingest_positives: bool, dataset: str):
    """Builds the per-detector ingest_fn for detect_batch(), keyed off `d.AWARENESS`
    (embedding-aware detectors get the cheap LanceDB-only insert, grag-aware ones get the real
    incremental reindex, corpus-unaware ones get nothing). Only ever fires for positive docs -
    negatives, if ingested at all, are already baked into the shared base+negatives snapshot.

    Each ingestion step is cached as its own version-store tag keyed by (AWARENESS, step index,
    dataset) - not by detector - since ingesting positive doc #i is identical work regardless of
    which detector triggers it (ingestion doesn't depend on a detector's own hops/top_n/alpha).
    The first detector of a given AWARENESS to reach step i does the real ingestion and tags it;
    every later detector of the same AWARENESS (including the dva-mode pass right after, which
    walks the same doc sequence) just restores that tag instead of repeating the real reindex -
    turns detector-count x doc-count real reindexes into just doc-count for the grag-aware ones.
    """
    if not ingest_positives or d.AWARENESS is None:
        return None
    ingest_one = d.embedding_connector.ingest_text if d.AWARENESS == "embedding" else d.graph_connector.ingest_document
    reload_one = d.embedding_connector.reload if d.AWARENESS == "embedding" else d.graph_connector.reload
    step = {"i": 0}

    def ingest_fn(doc, poisoned):
        if not poisoned:
            return
        i = step["i"]
        step["i"] += 1
        step_tag = _find_tag(f"benchmark_step_{d.AWARENESS}_{i}_{dataset}")
        if step_tag:
            v.restore(step_tag)
            reload_one()
        else:
            ingest_one(doc["new_text"])
            v.save(label=f"benchmark_step_{d.AWARENESS}_{i}_{dataset}")

    return ingest_fn


@log_time("benchmark")
async def run_benchmark(conf: dict) -> None:
    """Entry point for the whole benchmark stage: WVC setup, graphrag_under_fire, then per-dataset
    tuning + insertion/edit detection + charts."""
    printv("Running benchmark...", level="debug")

    detectors = conf["benchmark"]["detectors"]
    edit_detectors = conf["benchmark"]["detectors"]
    ingest_mode = conf["benchmark"].get("ingest_detected", "all")  # "positive" | "negative" | "none" | "all"

    # --- WVC setup: runs once before the dataset loop, only if enabled ---
    from scripts.wvc import PANWVCCorpus
    wvc_conf = conf["benchmark"].get("wvc", {})
    wvc = None
    if wvc_conf.get("enabled", False):
        wvc_corpus_path = wvc_conf.get("corpus_path", "datasets/pan-wikipedia-vandalism-corpus-2010")
        wvc_categories  = wvc_conf.get("categories") or None  # empty list → None → no filter
        wvc_category_pattern = wvc_conf.get("category_pattern") or None
        wvc_subset_size = wvc_conf.get("subset_size") or None
        wvc_reindex     = wvc_conf.get("reindex", False)

        wvc = PANWVCCorpus(wvc_corpus_path)
        if wvc_category_pattern:
            wvc.build_category_index()

        corpus_jsonl = Path(wvc_conf.get("data_root", "datasets/wvc")) / "corpus.jsonl"
        cat_label = wvc_category_pattern or (", ".join(wvc_categories) if wvc_categories else "all categories")

        set_active_corpus("wvc")  # switch graphrag to the wvc corpus dirs before checking index status

        if wvc_reindex:
            graphrag_output = Path("graphrag-api/graphrag/output/wvc")
            if graphrag_output.exists():
                shutil.rmtree(graphrag_output)
                printv("Cleared old GraphRAG WVC index for reindex.", level="info")

        if wvc_reindex or not corpus_jsonl.exists():
            printv(f"Building WVC corpus [{cat_label}]...", level="info")
            n_written = wvc.build_corpus_jsonl(
                output_path=corpus_jsonl, subset_size=wvc_subset_size,
                categories=wvc_categories, category_pattern=wvc_category_pattern,
            )
            printv(f"WVC corpus ready: {n_written} documents [{cat_label}]", level="info")
        else:
            printv(f"WVC corpus.jsonl exists — skipping rebuild [{cat_label}].", level="info")

        # GraphRAG's json loader expects a JSON array, not newline-delimited JSONL.
        sync_corpus_to_graphrag_input("wvc", corpus_jsonl)

        if wvc_reindex or not check_index_status()["is_complete"]:
            printv("Indexing WVC subset with GraphRAG...", level="info")
            await graphrag_init()
        else:
            printv("GraphRAG WVC index already complete — skipping indexing.", level="info")
            restart_graphrag_api()
    # --- end WVC setup ---

    for dataset in conf["benchmark"]["datasets"]:
        ensure_dataset_synced(dataset, conf)
        if conf["benchmark"]["versioning"]:
            _setup_benchmark_base_version(conf)

        n = conf["benchmark"].get("n", 10)
        val_size = conf["benchmark"].get("optimize_params", {}).get("val_size", 10)
        from scripts.compare_corpus import extract_diffs, extract_unchanged, load_jsonl
        clean_corpus_path = os.path.join("grag-data", "input", dataset, "corpus.jsonl")
        clean_records = load_jsonl(clean_corpus_path)

        sd_conf = conf["benchmark"].get("synthetic_dataset", {}) if dataset == "synthetic_dataset" else {}
        if dataset == "synthetic_dataset":
            # Synthetic fictional-world dataset (see scripts/detector.py's parametric-knowledge
            # thesis question). n/val_size are overridden per its own config block since the pool
            # is much smaller than wvc's.
            n = sd_conf.get("n", n)
            val_size = sd_conf.get("val_size", val_size)

        poisoned_corpus_path = os.path.join(conf["attack_config"]["poisoned_root"], dataset, "corpus.jsonl")
        if not os.path.isfile(poisoned_corpus_path):
            printv("No existing attack results found for dataset, please run attack pipeline first.", level="error")
            exit(1)
        printv("Existing attack results found for dataset, loading...", level="debug")
        poisoned_records = load_jsonl(poisoned_corpus_path)
        all_poisoned_diffs = extract_diffs(clean_records, poisoned_records)
        if dataset == "wvc" and wvc is not None:
            # LogicPoison poisons nearly this whole small, topically-coherent corpus (poisoning a
            # shared entity like a league or national team cascades across most player articles),
            # so "docs LogicPoison happened to skip" isn't a usable negative pool. Real WVC "regular"
            # (non-vandalism) edits are a genuine, separate source of clean examples instead.
            title_to_id = {r["title"]: int(r["_id"]) for r in clean_records}  # corpus.jsonl's _id is a string; extract_diffs' record_id is int - must match for base-index exclusion to work
            all_clean_diffs = [
                {**d, "record_id": title_to_id[d["article_title"]]}
                for d in wvc.get_diffs(change_type="clean", article_titles=set(title_to_id))
                if d["article_title"] in title_to_id
            ]
        elif dataset == "synthetic_dataset":
            # Same reasoning as wvc above, same fix: LogicPoison poisons the high-frequency
            # entities that recur across nearly this whole small, self-referential corpus (350
            # docs, 3 fictional nations), so "docs the attack happened to skip" is not a usable
            # negative pool (measured: 70 poisoned vs. 1 untouched). The clean class instead comes
            # from the LLM-authored "legit" half of diffs.jsonl (see
            # scripts/synthetic_dataset_generation/generate.py) - genuine edits, same
            # generation/length as the poisoned half used to be before the poisoned half switched
            # to this LogicPoison attack pass, so a detector still has to judge correctness rather
            # than just "was this text touched at all" (the confound extract_unchanged's
            # untouched-original comparison would have introduced instead).
            legit_diffs_path = sd_conf.get("diffs_path", os.path.join("grag-data", "input", dataset, "diffs.jsonl"))
            all_clean_diffs = [d for d in load_jsonl(legit_diffs_path) if not d.get("is_vandalism")]
        else:
            all_clean_diffs = extract_unchanged(clean_records, poisoned_records)
        #TODO: randomize with seed, dont only pick first n
        poisoned_diffs = all_poisoned_diffs[:n]
        clean_diffs = all_clean_diffs[:n]
        poisoned_val = all_poisoned_diffs[n:n + val_size]
        clean_val = all_clean_diffs[n:n + val_size]

        # Every doc scored anywhere this run (report split + optimization split) is excluded from
        # the base index build below, so no detector can find a test doc's own original already
        # sitting in the graph/embeddings it's being scored against.
        test_record_ids = {d["record_id"] for d in poisoned_diffs + clean_diffs + poisoned_val + clean_val}
        base_tag, base_with_negatives_tag = await _ensure_base_versions(
            dataset, conf, clean_corpus_path, test_record_ids,
            negative_diffs=clean_diffs + clean_val, build_with_negatives=ingest_mode in ("all", "negative"),
            exclude_test_docs=not (dataset == "wvc" and wvc is not None),
        )
        starting_tag = base_with_negatives_tag or base_tag
        v.restore(starting_tag)

        from scripts.detector import NLIScorer, calibrate_detector
        embedding_connector = EmbeddingConnector(corpus=dataset)
        graph_connector = GraphConnector(corpus=dataset)
        nli_scorer = NLIScorer()

        ingest_positives = ingest_mode in ("all", "positive")
        records_base = Path(conf["benchmark"].get("records_dir", "results/detector_records/"))
        records_dir = records_base / dataset

        # n-mode and dva-mode are optimized (and their detector instances built) independently -
        # a structural param like llm_as_a_judge's `variant` can genuinely differ between what's
        # best for each mode (see _optimize_detector_params), so a shared instance can't serve
        # both once optimize_params tunes per-mode.
        if conf["benchmark"]["detect_insertions"]:
            best_params_n = _optimize_detector_params(
                conf, dataset, set(detectors), all_poisoned_diffs, all_clean_diffs, use_dva=False,
                reserved=n, embedding_connector=embedding_connector, graph_connector=graph_connector, nli_scorer=nli_scorer,
                val_size=val_size,
            )
            objects_n = _build_detector_objects(conf, embedding_connector, graph_connector, nli_scorer, best_params_n)
            active_n = [o for o in objects_n if type(o).__name__ in detectors]
            printv(f"Running n-mode benchmark with {len(active_n)} detectors, {len(poisoned_diffs)} poisoned diffs, and {len(clean_diffs)} clean diffs.", level="info")
            tick = make_progress_ticker(len(active_n)*(len(clean_diffs)+len(poisoned_diffs)))
            for d in active_n:
                out_path = records_dir / f"{getattr(d, 'name', type(d).__name__)}_insertions.json"
                if out_path.exists():
                    # Already scored by an earlier, since-interrupted run - skip rather than redo,
                    # same reasoning as optimize_all's cache. Delete the file (or the whole
                    # records_dir) to force a redo if config/inputs changed since.
                    printv(f"{out_path.name} already exists - skipping (delete it to force a redo).", level="info")
                    continue
                v.restore(starting_tag)
                embedding_connector.reload()
                graph_connector.reload()
                try:
                    if hasattr(d, "calibrate"):
                        # Calibrated against the common starting_tag graph, before this detector's own
                        # in-batch ingestion runs - same null distribution every detector would see.
                        calibrate_detector(d, clean_val, use_dva=False, poisoned_docs=poisoned_val)
                    d.detect_batch(clean_diffs, poisoned_diffs, ingest_fn=_make_ingest_fn(d, ingest_positives, dataset), progress_cb=tick)
                    d.save_to_file(out_path)
                finally:
                    v.restore(starting_tag)
        if conf["benchmark"]["detect_edits"]:
            best_params_dva = _optimize_detector_params(
                conf, dataset, set(edit_detectors), all_poisoned_diffs, all_clean_diffs, use_dva=True,
                reserved=n, embedding_connector=embedding_connector, graph_connector=graph_connector, nli_scorer=nli_scorer,
                val_size=val_size,
            )
            objects_dva = _build_detector_objects(conf, embedding_connector, graph_connector, nli_scorer, best_params_dva)
            # Only detectors that actually score edits differently. The rest inherit
            # detector.score_dva, which just calls score_n on new_text - so a dva pass would
            # recompute their n-mode score and record a bit-for-bit identical AUC under an
            # "_edits" name, doubling the run for no information. See detector.has_dva_strategy().
            active_dva = [o for o in objects_dva
                          if type(o).__name__ in edit_detectors and type(o).has_dva_strategy()]
            skipped_dva = sorted({type(o).__name__ for o in objects_dva
                                  if type(o).__name__ in edit_detectors and not type(o).has_dva_strategy()})
            if skipped_dva:
                printv(f"dva mode: skipping {', '.join(skipped_dva)} - no score_dva of their own, "
                       f"their edit scores would duplicate their insertion scores exactly.", level="info")
            tick = make_progress_ticker(len(active_dva)*n*2)
            for d in active_dva:
                out_path = records_dir / f"{getattr(d, 'name', type(d).__name__)}_edits.json"
                if out_path.exists():
                    printv(f"{out_path.name} already exists - skipping (delete it to force a redo).", level="info")
                    continue
                v.restore(starting_tag)
                embedding_connector.reload()
                graph_connector.reload()
                try:
                    if hasattr(d, "calibrate"):
                        calibrate_detector(d, clean_val, use_dva=True, poisoned_docs=poisoned_val)
                    d.detect_batch(clean_diffs, poisoned_diffs, use_dva=True, ingest_fn=_make_ingest_fn(d, ingest_positives, dataset), progress_cb=tick)
                    d.save_to_file(out_path)
                finally:
                    v.restore(starting_tag)

        # Document-length sweep - disabled (needs benchmark.wvc.enabled too, neither is on by
        # default) and superseded by the per-dataset comparison above. Commented out, not deleted:
        # sweep_conf = conf["benchmark"].get("document_length_sweep", {})
        # if sweep_conf.get("enabled", False):
        #     if wvc is None:
        #         printv("document_length_sweep requires benchmark.wvc.enabled=true — skipping.", level="warning")
        #     else:
        #         await _run_length_sweep(wvc, edit_detectors, sweep_conf, Path(conf["benchmark"]["out_dir"]), records_base / "wvc_length_sweep")

    # Runs after the dataset loop above - _run_logicpoison_no_negatives reuses wvc's just-tuned
    # best_params (wvc_n.json/wvc_dva.json), which only exist once that loop has run.
    await _run_graphrag_under_fire(conf)
    await _run_logicpoison_no_negatives(conf)


def _setup_benchmark_base_version(conf: dict) -> None:
    """Ensures a clean-corpus base version exists to roll back to, deleting any stale non-base
    versions left over from a previous run."""
    base_corpus_size = conf["benchmark"]["base_corpus_size"] or 0.5
    base_label = f"benchmark_base_{base_corpus_size}"
    existing_versions = v.list_versions()
    base_version = next((ver for ver in existing_versions if ver.get("message") == base_label), None)
    if base_version is not None:
        printv(f"Base version exists: {base_label}", level="info")
    printv("Cleaning up non-base versions...", level="info")
    for ver in existing_versions:
        if ver.get("message") != base_label:
            printv(f"Deleting version: {ver['tag']} ({ver.get('message', '')})", level="debug")
            try:
                subprocess.run(["git", "tag", "-d", ver["tag"]], check=True, cwd=".")
            except subprocess.CalledProcessError as e:
                printv(f"Warning: Could not delete tag {ver['tag']}: {e}", level="warning")
    if not base_version:
        printv("Base version does not exist. Creating new base...", level="info")
        #TODO: create a reproducible random and select basis
        v.save(label=base_label)
        printv(f"Benchmark base version created and saved: {base_label}", level="info")


async def _run_graphrag_under_fire(conf: dict) -> None:
    """Benchmarks each graphrag_under_fire attack type as a free-standing pool of fabricated
    documents against the clean corpus - insertion-only, since there's no paired original to diff
    an edit against (see setup_dataset.py::build_graphrag_under_fire_dataset)."""
    guf_conf = conf["benchmark"].get("graphrag_under_fire", {})
    if not guf_conf.get("enabled", False):
        return

    dataset = "graphrag_under_fire"
    # Poisoned pool here (~200/attack type) is much smaller than hotpotqa/musique's, so these
    # override the global n/val_size to leave enough for the optimize_params validation slice.
    n = guf_conf.get("n", conf["benchmark"].get("n", 10))
    val_size = guf_conf.get("val_size") or conf["benchmark"].get("optimize_params", {}).get("val_size", 10)
    poisoned_dir = Path(guf_conf.get("poisoned_dir", f"results/poisoned_data/{dataset}"))
    clean_corpus_path = Path("grag-data") / "input" / dataset / "corpus.jsonl"

    from scripts.compare_corpus import load_jsonl
    from scripts.detector import NLIScorer, calibrate_detector

    ensure_dataset_synced(dataset, conf)

    def _as_diffs(records: list[dict], poisoned: bool) -> list[dict]:
        return [
            {"record_id": r["_id"], "original_text": "" if poisoned else r["text"], "new_text": r["text"]}
            for r in records
        ]

    all_clean_diffs = _as_diffs(load_jsonl(clean_corpus_path), poisoned=False)
    clean_diffs = all_clean_diffs[:n]
    clean_val = all_clean_diffs[n:n + val_size]

    ingest_mode = conf["benchmark"].get("ingest_detected", "all")
    ingest_positives = ingest_mode in ("all", "positive")

    records_base = Path(conf["benchmark"].get("records_dir", "results/detector_records/")) / dataset
    detector_names = set(conf["benchmark"]["detectors"])

    # Base index built ONCE, shared across every attack type - the clean corpus (and so what
    # needs excluding/held out) is identical regardless of which attack's poisoned docs get
    # scored against it; the fabricated attack docs never touch the clean corpus's own id
    # namespace either way. Building a separate index per attack type repeated identical
    # extract_graph/community_reports work three times over for no reason.
    test_record_ids = {d["record_id"] for d in clean_diffs + clean_val}
    base_tag, base_with_negatives_tag = await _ensure_base_versions(
        dataset, conf, str(clean_corpus_path), test_record_ids,
        negative_diffs=clean_diffs + clean_val, build_with_negatives=ingest_mode in ("all", "negative"),
    )
    starting_tag = base_with_negatives_tag or base_tag
    v.restore(starting_tag)

    embedding_connector = EmbeddingConnector(corpus=dataset)
    graph_connector = GraphConnector(corpus=dataset)
    nli_scorer = NLIScorer()

    for attack_type in guf_conf.get("attack_types", ["direct", "indirect", "enhanced"]):
        poison_path = poisoned_dir / f"corpus_{attack_type}.jsonl"
        if not poison_path.is_file():
            printv(f"graphrag_under_fire: missing {poison_path} - run "
                   f"setup_dataset.build_graphrag_under_fire_dataset() first. Skipping '{attack_type}'.", level="warning")
            continue
        all_poisoned_diffs = _as_diffs(load_jsonl(poison_path), poisoned=True)
        poisoned_diffs = all_poisoned_diffs[:n]
        poisoned_val = all_poisoned_diffs[n:n + val_size]

        # Optuna tuning stays per-attack-type (best params can genuinely differ by attack
        # strategy) even though the underlying index is shared.
        best_params = _optimize_detector_params(
            conf, f"{dataset}_{attack_type}", detector_names, all_poisoned_diffs, all_clean_diffs, use_dva=False,
            reserved=n, embedding_connector=embedding_connector, graph_connector=graph_connector, nli_scorer=nli_scorer,
            val_size=val_size,
        )
        objects = _build_detector_objects(conf, embedding_connector, graph_connector, nli_scorer, best_params)
        active_detectors = [o for o in objects if type(o).__name__ in detector_names]

        printv(f"graphrag_under_fire[{attack_type}]: {len(active_detectors)} detectors, "
               f"{len(poisoned_diffs)} poisoned diffs, {len(clean_diffs)} clean diffs.", level="info")
        tick = make_progress_ticker(len(active_detectors) * (len(clean_diffs) + len(poisoned_diffs)))
        for d in active_detectors:
            v.restore(starting_tag)
            embedding_connector.reload()
            graph_connector.reload()
            try:
                if hasattr(d, "calibrate"):
                    # graphrag_under_fire is n-mode only (see this function's docstring), so the
                    # null is always built from absolute values, never from edit regions.
                    calibrate_detector(d, clean_val, use_dva=False, poisoned_docs=poisoned_val)
                d.detect_batch(clean_diffs, poisoned_diffs, ingest_fn=_make_ingest_fn(d, ingest_positives, dataset), progress_cb=tick)
                d.save_to_file(records_base / f"{getattr(d, 'name', type(d).__name__)}_{attack_type}.json")
            finally:
                v.restore(starting_tag)


async def _run_logicpoison_no_negatives(conf: dict) -> None:
    """Scores LogicPoison-poisoned musique/hotpotqa as positives-only (both n and dva mode) -
    unlike wvc, these datasets have no real-clean-edit donor corpus, so there's no usable negative
    set and only tp/fn/recall are meaningful (see detector.summarize_positive_only()). Reuses
    wvc's tuned params instead of running its own Optuna study - `benchmark.wvc`'s optimize_params
    run is treated as the single source of truth for structural params across datasets."""
    datasets = conf["benchmark"].get("logicpoison_only_datasets", [])
    if not datasets:
        return

    from scripts.compare_corpus import extract_diffs, load_jsonl
    from scripts.detector import NLIScorer, calibrate_detector
    from scripts.optimize import load_best_params

    n = conf["benchmark"].get("n", 10)
    best_params_dir = Path(conf["benchmark"].get("optimize_params", {}).get("best_params_dir", "results/detector_best_params/"))
    best_params_n = load_best_params(best_params_dir / "wvc_n.json")
    best_params_dva = load_best_params(best_params_dir / "wvc_dva.json")
    detector_names = set(conf["benchmark"]["detectors"])
    records_base = Path(conf["benchmark"].get("records_dir", "results/detector_records/"))

    for dataset in datasets:
        ensure_dataset_synced(dataset, conf)
        set_active_corpus(dataset)
        if not check_index_status()["is_complete"]:
            await graphrag_init()
        else:
            restart_graphrag_api()

        poisoned_path = os.path.join(conf["attack_config"]["poisoned_root"], dataset, "corpus.jsonl")
        if not os.path.isfile(poisoned_path):
            printv(f"logicpoison_only: no poisoned data for '{dataset}' at {poisoned_path} - skipping.", level="warning")
            continue
        clean_records = load_jsonl(os.path.join("grag-data", "input", dataset, "corpus.jsonl"))
        poisoned_diffs = extract_diffs(clean_records, load_jsonl(poisoned_path))[:n]
        if not poisoned_diffs:
            continue
        # There are no clean *edit pairs* for these datasets (that's why they're positives-only),
        # but the unpoisoned corpus documents themselves are exactly the right null for a detector
        # calibrated on "what does clean look like" - gtopo. Without this it used to be handed [],
        # leaving its null empty so every score collapsed to one constant and recall was a
        # fabricated 0.0. Excludes the documents the poisoned diffs were derived from.
        _poisoned_ids = {d.get("record_id") for d in poisoned_diffs}
        calibration_docs = [
            {"new_text": r.get("text", ""), "original_text": r.get("text", ""), "record_id": r.get("id")}
            for r in clean_records if r.get("id") not in _poisoned_ids and r.get("text")
        ][:100]

        embedding_connector = EmbeddingConnector(corpus=dataset)
        graph_connector = GraphConnector(corpus=dataset)
        nli_scorer = NLIScorer()
        records_dir = records_base / dataset
        summary: dict[str, dict] = {}

        # Thresholds come from wvc's Optuna studies and are used AS GIVEN here. With
        # optimize_threshold on, detector.finalize() re-derives the cut from the records it just
        # scored - but these datasets are positives-only, so _optimal_threshold() cannot run a ROC
        # and falls back to (min+max)/2 of the detector's own scores. Every "recall" then meant
        # "fraction scoring above the midpoint of this detector's range", which is a property of
        # each score distribution's shape rather than a detection rate, and is not comparable
        # between detectors (measured: perplexity's heavy tail put its midpoint above almost every
        # document, giving recall 0.03 on hotpotqa). A threshold chosen on wvc's labelled
        # validation split is a real operating point, so it is kept.
        threshold_conf = {**conf, "benchmark": {**conf["benchmark"], "optimize_threshold": False}}
        for use_dva, params in ((False, best_params_n), (True, best_params_dva)):
            objects = _build_detector_objects(threshold_conf, embedding_connector, graph_connector, nli_scorer, params)
            active = [o for o in objects if type(o).__name__ in detector_names]
            if use_dva:
                # Same reason as the wvc/dva branch above: a detector without its own score_dva
                # would re-score new_text through score_n and record an identical result under a
                # "_dva" name. See detector.has_dva_strategy().
                active = [o for o in active if type(o).has_dva_strategy()]
            mode = "dva" if use_dva else "n"
            # rand/rand1/perplexity_filtering have no structural params, so they are not in the
            # tunable set and no wvc study ever produced a threshold for them - they run on their
            # constructor defaults (rand 0.5, perplexity abs 40.0 / ratio 1.05). Those are fixed
            # constants rather than anything fitted to this corpus, so say so instead of letting
            # their recall read like a tuned operating point.
            untuned = [type(o).__name__ for o in active if type(o).__name__ not in params]
            if untuned:
                printv(f"logicpoison_only[{dataset}/{mode}]: no tuned threshold for "
                       f"{', '.join(sorted(untuned))} - using constructor defaults.", level="warning")
            tick = make_progress_ticker(len(active) * len(poisoned_diffs))
            for d in active:
                if hasattr(d, "calibrate"):
                    # Mode matters here: this loop runs both n and dva, and a dva-aware null has
                    # to be built through the same score path. calibration_docs are unedited
                    # documents (new_text == original_text), so a detector that localises to the
                    # changed region finds nothing to measure in dva mode - the `calibrated` guard
                    # below is what catches that and skips it rather than reporting a fake recall.
                    calibrate_detector(d, calibration_docs, use_dva=use_dva)
                    if not getattr(d, "calibrated", True):
                        # Scoring uncalibrated yields one identical constant for every document,
                        # i.e. a recall that looks like a measurement but isn't. Skip and say so.
                        printv(f"logicpoison_only[{dataset}]: {getattr(d, 'name', type(d).__name__)} "
                               f"could not be calibrated ({len(calibration_docs)} clean docs) - skipping.",
                               level="warning")
                        continue
                d.detect_batch([], poisoned_diffs, use_dva=use_dva, progress_cb=tick)
                d.save_to_file(records_dir / f"{getattr(d, 'name', type(d).__name__)}_{mode}.json")
                positive_only = d.summarize_positive_only(use_dva=use_dva)
                summary[f"{getattr(d, 'name', type(d).__name__)}_{mode}"] = positive_only

        from scripts.detector import write_json
        write_json(summary, records_dir / "positive_only_summary.json")
        printv(f"logicpoison_only[{dataset}]: {summary}", level="info")


