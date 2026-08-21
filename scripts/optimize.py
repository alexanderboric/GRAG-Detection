"""Hyperparameter search for detector constructor params that change what score_n()/score_dva()
computes (k_neighbors, hops, top_n, k...), run once per dataset on data disjoint from final
reporting. Caller picks which mode (n or dva) to optimize against via `use_dva`.
"""

import json
import logging
import math
import threading
from pathlib import Path
from typing import Callable

import openai
import optuna
from tqdm import tqdm

from scripts.detector import (
    calibrate_detector,
    embedding_outlier,
    gnlic,
    grag_as_a_judge,
    gtopo,
    gtopo2,
    llm_as_a_judge,
    nli_contradiction,
    prompt_variant_count,
    write_json,
)

logger = logging.getLogger(__name__)

# Share of documents a trial may score through its no-evidence fallback before the configuration
# is rejected outright (see objective()). Half is deliberate: past that, the headline number is
# mostly the fallback detector's, not this one's.
_MAX_FALLBACK_RATE = 0.5
# Returned instead of an AUC by a rejected trial. Below every attainable AUC (0.0 = perfectly
# inverted, which is still a real measurement), so a rejected config can never win a study - but
# study.best_trial still exists if every trial is rejected, and the caller logs the result.
_REJECTED = -1.0
# Optuna's own per-trial INFO logging is silenced in favour of the tqdm progress bar
# below (_progress_callback), which shows the same "trial N, current best" info without
# the one-line-per-trial log spam.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_nli_contradiction(trial: optuna.Trial, *, embedding_connector, nli_scorer, **_) -> nli_contradiction:
    k_neighbors = trial.suggest_int("k_neighbors", 1, 20)
    return nli_contradiction(embedding_connector, nli_scorer, k_neighbors=k_neighbors, optimize_threshold=True)


# def _build_grag_contradiction(trial: optuna.Trial, *, graph_connector, nli_scorer, embedding_connector=None, **_) -> grag_contradiction:
#     hops = trial.suggest_int("hops", 1, 3)
#     top_n = trial.suggest_int("top_n", 1, 10)
#     return grag_contradiction(graph_connector, nli_scorer, embedding_connector=embedding_connector,
#                  hops=hops, top_n=top_n, optimize_threshold=True)


def _build_embedding_outlier(trial: optuna.Trial, *, embedding_connector, **_) -> embedding_outlier:
    k = trial.suggest_int("k", 1, 20)
    return embedding_outlier(embedding_connector, k=k, optimize_threshold=True)


def _build_gnlic(trial: optuna.Trial, *, graph_connector, nli_scorer, embedding_connector=None, **_) -> gnlic:
    # Retrieval goes through GraphRAG's own local_search (see GraphConnector.query_context), not a
    # hops/top_n hand-traversal. `method` is NOT tuned: measured coverage was ~120/120 documents for
    # "local" against ~21/120 for "global" (graphrag scores all community reports at 0 relevance
    # against a document-as-query and returns its canned refusal). That is not a trade-off for an
    # optimizer to make - and with the no-evidence fallback in place "global" would now look healthy
    # while ~85% of its score came from the fallback rather than the graph.
    alpha = trial.suggest_categorical("alpha", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # community_level feeds query_context()'s community reports (same knob as gtopo's `level`,
    # same GraphRAG communities.parquet) - previously hardcoded to the constructor default (2)
    # rather than searched, unlike every other structural param this detector has. Same range as
    # gtopo's grid (see _build_gtopo) for the same reason: coarser levels always exist, finer ones
    # may not on a small graph.
    community_level = trial.suggest_categorical("community_level", [0, 1, 2, 3])
    # k_neighbors only matters on the no-evidence fallback path (_embedding_nli_max, same helper
    # nli_contradiction uses) - previously hardcoded to the constructor default (3). Same range as
    # nli_contradiction's grid for the same reason: it's the same fallback computation.
    k_neighbors = trial.suggest_int("k_neighbors", 1, 20)
    return gnlic(graph_connector, nli_scorer, embedding_connector=embedding_connector,
                 alpha=alpha, community_level=community_level, k_neighbors=k_neighbors,
                 optimize_threshold=True)


def _build_gtopo(trial: optuna.Trial, *, graph_connector, **_) -> gtopo:
    # level=0 added after graphrag_under_fire's study rejected every one of level in {1,2,3}
    # (communities.parquet only has levels 0/1 at all for its 208-doc graph - level 2/3 don't
    # exist, so every pair abstained there regardless of q). level=0 is GraphRAG's coarsest level
    # and always exists, but re-running with it included confirmed the problem is coverage, not
    # granularity: even at level=0 only 58/485 entities have any community, so the best trial
    # still fell back on 89% of documents (vs. 92-100% for 1-3) - short of the <=50% the fallback
    # guard requires, so gtopo's GUF entry stays unoptimized (see optimize_detector_params's
    # _REJECTED handling) rather than recording params that don't actually use its own evidence.
    # Kept in the grid anyway (measurably better than 1-3, and wvc's bigger graph still tunes to
    # level=2 there, so this costs it nothing) - the real fix would be a source of communities
    # that doesn't need Leiden to already have a big-enough graph to work with.
    level = trial.suggest_categorical("level", [0, 1, 2, 3])
    q = trial.suggest_categorical("q", [0.7, 0.8, 0.9, 1.0])
    return gtopo(graph_connector, level=level, q=q, optimize_threshold=True)


# Candidate signal subsets for gtopo2, not the full 15-subset powerset: the grid is multiplied by
# level x q (12), and a 180-point space cannot be exhausted inside n_trials=30, which is what makes
# the other studies terminate early and deterministically. These four are the ones the wvc report
# split distinguishes - all-four (the current behaviour), the two that beat it per mode, and the
# pair-only set that drops the whole-document signal.
GTOPO2_SIGNAL_SETS = ["all", "dist+cohesion", "blockodds", "dist+support+blockodds"]


def _build_gtopo2(trial: optuna.Trial, *, graph_connector, **_) -> gtopo2:
    # Same two params as v1's grid, deliberately: gtopo2 also exposes dist_cap/max_pairs/
    # max_context, but level x q is 12 points and n_trials is 30, so adding a third axis would
    # overflow the budget and leave GridSampler's space unexhausted (the studies currently finish
    # early precisely because they exhaust it). Matching v1's grid also keeps the two detectors'
    # tuning effort comparable, which is the point of running them side by side.
    level = trial.suggest_categorical("level", [1, 2, 3])
    q = trial.suggest_categorical("q", [0.7, 0.8, 0.9, 1.0])
    signals = trial.suggest_categorical("signals", GTOPO2_SIGNAL_SETS)
    return gtopo2(graph_connector, level=level, q=q, signals=signals, optimize_threshold=True)


def _build_llm_as_a_judge(trial: optuna.Trial, **_) -> llm_as_a_judge:
    variant = trial.suggest_categorical("variant", list(range(prompt_variant_count("llm_as_a_judge"))))
    return llm_as_a_judge(variant=variant, optimize_threshold=True)


def _build_grag_as_a_judge(trial: optuna.Trial, *, graph_connector, **_) -> grag_as_a_judge:
    # `method` fixed to "local" for the same measured reason as _build_gnlic.
    variant = trial.suggest_categorical("variant", list(range(prompt_variant_count("grag_as_a_judge"))))
    return grag_as_a_judge(graph_connector, variant=variant, optimize_threshold=True)


# detectors with structural (score-changing) params worth searching. perplexity_filtering
# (model_name) and rand/rand1 (no structural params at all) are deliberately
# excluded - either nothing to tune, or every trial would mean reloading a whole model for no
# real gain in this setup. llm_as_a_judge/grag_as_a_judge tune `variant` - which of
# prompts.jsonl's system-prompt strategies to use - now that build_prompts.py generates
# multiple genuinely different candidates per detector/mode instead of a single fixed prompt.
#
# Each entry also carries its full discrete search space (name -> list of exact values
# `suggest_int` is allowed to hit): every one of these grids has <=30 combinations, well
# under n_trials, so GridSampler (see optimize_detector_params) is used instead of the
# default TPE sampler - TPE has no dedup guard and, on a search space this small, reliably
# wastes trials re-testing params it already tried (e.g. two of 8 completed
# grag_contradiction trials on hotpotqa landed on the exact same hops=1/top_n=15). Grid
# search covers the space exhaustively with zero duplicates, and finishes once the grid is
# exhausted rather than continuing to (re)sample for the full n_trials.
_TUNABLE: dict[str, tuple[Callable, dict[str, list]]] = {
    # Ordered by actual dependency on the (frequently degraded/down) embedding endpoint, not just
    # "graph vs completion" - dict order is what optimize_all() processes in (via
    # tunable_detector_names()), so this keeps every detector that DOESN'T need embeddings from
    # getting stuck behind ones that do. gnlic and grag_as_a_judge both route through
    # GraphConnector.query_context() -> GraphRAG's local_search, which does its own internal
    # embedding-based retrieval step before it ever reaches completion - confirmed directly
    # (isolated test calls: completion endpoint answered in 0.6s, embedding endpoint timed out
    # 3/3 times) that this, not the completion model, is what was actually hanging the whole
    # benchmark. gtopo is pure local graph math (no LLM/embedding calls at all). llm_as_a_judge
    # is pure completion (no graph/embedding connector). nli_contradiction only needs embeddings
    # in n mode - score_dva() bypasses them entirely (see _DVA_INVARIANT below), so it's
    # effectively embedding-free for the dva-only runs this project currently does.
    # "grag_contradiction": (_build_grag_contradiction, {"hops": [1, 2, 3], "top_n": list(range(1, 11))}),
    # level=0 included alongside 1-3 - see _build_gtopo's comment for why.
    "gtopo": (_build_gtopo, {"level": [0, 1, 2, 3], "q": [0.7, 0.8, 0.9, 1.0]}),
    # Directly after v1 and before the embedding-dependent entries: gtopo2 is also pure local
    # graph math (no LLM or embedding calls), so it belongs in the same no-network group.
    "gtopo2": (_build_gtopo2, {"level": [1, 2, 3], "q": [0.7, 0.8, 0.9, 1.0],
                               "signals": GTOPO2_SIGNAL_SETS}),
    "llm_as_a_judge": (_build_llm_as_a_judge, {"variant": list(range(prompt_variant_count("llm_as_a_judge")))}),
    "nli_contradiction": (_build_nli_contradiction, {"k_neighbors": list(range(1, 21))}),
    "gnlic": (_build_gnlic, {"alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                             "community_level": [0, 1, 2, 3], "k_neighbors": list(range(1, 21))}),
    "grag_as_a_judge": (_build_grag_as_a_judge, {"variant": list(range(prompt_variant_count("grag_as_a_judge")))}),
    "embedding_outlier": (_build_embedding_outlier, {"k": list(range(1, 21))}),
}


# Detectors whose search-space param(s) have zero effect on score_dva() specifically - confirmed
# by reading the code: nli_contradiction.score_dva() calls
# self.nli_scorer.contradiction_prob(doc["original_text"], doc["new_text"]) directly and never
# touches k_neighbors/embedding_connector at all (unlike score_n(), which does). Since score_n() is
# deterministic given fixed params and there's no other source of run-to-run noise, every trial in
# a dva-mode (use_dva=True) study for these detectors would compute a byte-for-bit identical
# score on every document - a real duplicate, not just a similar result. optimize_detector_params
# caps these to a single trial when use_dva=True instead of burning the whole grid for no
# additional information. Empty for n-mode studies, where these same params DO matter.
#
# embedding_outlier joins it for the same reason: score_dva() now embeds doc["original_text"]/
# doc["new_text"] directly and diffs those two vectors - it never calls nearest()/
# corpus_distance_stats(), so `k` (the only thing _build_embedding_outlier's grid searches) has
# zero effect on a dva-mode trial's score.
_DVA_INVARIANT = {"nli_contradiction", "embedding_outlier"}


def tunable_detector_names() -> list[str]:
    return list(_TUNABLE.keys())


# Detector name -> class, so optimize_all can ask whether a dva-mode study is worth running at all
# without having to build the detector (which needs a live trial and real connectors).
_CLASSES = {
    "gtopo": gtopo,
    "gtopo2": gtopo2,
    "llm_as_a_judge": llm_as_a_judge,
    "nli_contradiction": nli_contradiction,
    "gnlic": gnlic,
    "grag_as_a_judge": grag_as_a_judge,
    "embedding_outlier": embedding_outlier,
}


def _has_dva_strategy(detector_name: str) -> bool:
    cls = _CLASSES.get(detector_name)
    # Unknown names default to True so a newly added detector is tuned rather than silently skipped.
    return cls.has_dva_strategy() if cls else True


def search_space(detector_name: str) -> dict[str, list]:
    """The parameter grid currently searched for `detector_name` ({} if it isn't tuned).

    Exposed so callers can drop params that a *previous* run tuned but the current search space no
    longer contains. best_params files are reused across runs (force_reoptimize=false), and the
    merge is by key, so a param removed from the grid would otherwise be silently reinstated from
    cache forever - e.g. graphrag_under_fire_*_n.json still carries method="global", which is
    exactly the setting that was removed for retrieving only ~21/120 documents.
    """
    entry = _TUNABLE.get(detector_name)
    return dict(entry[1]) if entry else {}


def _trial_history_callback(history_path: Path):
    # appended to after every single trial (not just at the end), so a study can be
    # interrupted/inspected mid-run to see how the running-best value moves with trial
    # count - the whole point being to answer "how many trials do we actually need?"
    # without waiting for a full study to finish first.
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("", encoding="utf-8")  # fresh file per study

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_trial
        record = {
            "trial": trial.number,
            "value": trial.value,
            "params": trial.params,
            "best_value_so_far": best.value,
            "best_params_so_far": best.params,
            "best_threshold_so_far": best.user_attrs.get("threshold"),
            "best_threshold_edit_so_far": best.user_attrs.get("threshold_edit"),
        }
        # Re-create the directory right before every append, not just once at study start - a
        # single slow trial (confirmed: one gnlic trial took 2h48m fighting through connection
        # resets and t429 rate-limit retries) leaves a long window where something external could
        # remove it, and losing this callback's write took the whole process down with an
        # uncaught FileNotFoundError - a trial that already paid the real cost of a successful
        # score must not be thrown away by a bug in recording it.
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return callback


def _progress_callback(name: str, n_trials: int):
    # same tqdm bar style as graphrag_manager.py's indexing progress - one line per
    # detector's study, advancing per trial, with the running-best AUC in the postfix
    # so you can watch convergence live instead of only reading it back from the
    # trial-history file afterwards.
    bar = tqdm(total=n_trials, desc=f"optimize({name})", ncols=80)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        bar.set_postfix(best_auc=f"{study.best_value:.3f}")
        bar.update(1)
        if bar.n >= bar.total:
            bar.close()

    return callback


def optimize_detector_params(
    name: str,
    clean_val: list[dict],
    poisoned_val: list[dict],
    *,
    n_trials: int = 20,
    embedding_connector=None,
    graph_connector=None,
    nli_scorer=None,
    history_path: str | Path | None = None,
    use_dva: bool = False,
) -> dict:
    """Runs an Optuna study over `name`'s tunable constructor params against score_n() by
    default; pass use_dva=True to score dva mode instead - matters for llm_as_a_judge/
    grag_as_a_judge specifically, since n and dva draw from disjoint prompts.jsonl pools
    (variant N's insertion prompt and variant N's edit prompt are unrelated strategies, only
    paired by list position - see scripts/build_prompts.py), so tuning against the wrong mode
    picks a variant that's good at a task it won't actually run.
    Each trial builds a fresh detector with optimize_threshold=True so its threshold is
    already optimal for that trial's params. `history_path`, if given, gets one JSON line
    appended per trial so best-so-far can be plotted afterwards.
    Returns {} if `name` has nothing tunable, else {"params", "threshold", "threshold_edit", "auc"}.
    """
    entry = _TUNABLE.get(name)
    if entry is None:
        return {}
    builder, search_space = entry
    if not clean_val or not poisoned_val:
        logger.warning("optimize_detector_params(%s): empty validation split, skipping.", name)
        return {}

    def objective(trial: optuna.Trial) -> float:
        det = builder(
            trial,
            embedding_connector=embedding_connector,
            graph_connector=graph_connector,
            nli_scorer=nli_scorer,
        )
        if hasattr(det, "calibrate"):
            # Generic hook for detectors (gtopo) that need a clean null distribution before
            # scoring. Pragmatic for now: calibrates on the same clean_val it's then scored
            # against (mild self-reference - the null distribution includes the points being
            # judged against it), not a separate held-out clean sample. Fine for a first pass;
            # revisit if gtopo's numbers need to be trusted more precisely.
            # use_dva must reach it: the study below scores in that mode, and gtopo2's null is
            # mode-specific, so calibrating n-mode and then scoring dva would tune every trial
            # against a null built by a different code path.
            calibrate_detector(det, clean_val, use_dva=use_dva, poisoned_docs=poisoned_val)
        # Per-document bar nested under the per-trial one - without it, a slow/stuck trial gives
        # zero visibility (nothing is written anywhere) until it either finishes or crashes.
        # Lock guards doc_bar.update() since parallelizable detectors call progress_cb
        # concurrently from detect_batch()'s ThreadPoolExecutor - tqdm.update() isn't thread-safe.
        doc_bar = tqdm(total=len(clean_val) + len(poisoned_val), desc=f"  {name} trial {trial.number}", ncols=80, leave=False)
        doc_bar_lock = threading.Lock()

        def _tick_doc_bar():
            with doc_bar_lock:
                doc_bar.update(1)

        det.detect_batch(clean_val, poisoned_val, use_dva=use_dva, progress_cb=_tick_doc_bar)
        doc_bar.close()
        auc = det.roc_auc(use_dva=use_dva)
        trial.set_user_attr("threshold", det.threshold_n)
        trial.set_user_attr("threshold_edit", det.threshold_dva)

        # A configuration that reached its AUC without its own evidence source on most documents
        # is not a valid instance of this detector, however well it happened to score. Measured
        # case: gtopo's grid offered level=3, which on wvc maps only 43 of 2307 entities
        # (1.9% node coverage, 9 block rates), so pair_surprise returned None almost everywhere,
        # inc/novlev abstained on 197/200 documents, and _p()'s uninformative 0.5 made the score
        # near-constant - yet it won the dva study on validation noise. Rejecting by coverage
        # fixes that generally instead of hand-pruning `level`, and applies to any grag-aware
        # detector whose params can quietly disable its evidence source.
        coverage = det.coverage(use_dva=use_dva) if hasattr(det, "coverage") else None
        if coverage and coverage.get("n_total"):
            rate = coverage["n_fallback"] / coverage["n_total"]
            trial.set_user_attr("fallback_rate", rate)
            if rate > _MAX_FALLBACK_RATE:
                logger.warning(
                    "%s trial %d (%s): fell back on %.0f%% of documents (auc %.3f) - "
                    "rejecting, the params disable this detector's own evidence source.",
                    name, trial.number, trial.params, 100 * rate, auc,
                )
                return _REJECTED
        return auc

    if use_dva and name in _DVA_INVARIANT:
        # every point in the grid scores identically under score_dva() for this detector - see
        # _DVA_INVARIANT. One trial is enough; the other grid_size-1 would be pure duplicates.
        n_trials = 1

    callbacks = [_progress_callback(name, n_trials)]
    if history_path:
        callbacks.append(_trial_history_callback(Path(history_path)))

    grid_size = math.prod(len(values) for values in search_space.values())
    if grid_size <= n_trials:
        # budget covers the whole discrete space - exhaustive search dominates TPE here:
        # guaranteed to find the true best combo, zero wasted duplicate evaluations. TPE
        # has no dedup guard and, worse, nothing to gain from ever resampling a point:
        # score() is deterministic given fixed params, so a repeat trial is 100% wasted.
        sampler = optuna.samplers.GridSampler(search_space)
    else:
        # budget can't cover the space - fall back to Bayesian search (TPE) so the
        # limited trials get spent adaptively on promising regions instead of an
        # arbitrary (grid or random) subset.
        sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", study_name=f"{name}_optimization", sampler=sampler)
    # catch: a persistent API failure (retries in retry_api_call exhausted) fails just this one
    # trial, not the whole multi-hour study - important for an unattended run.
    # n_jobs=1 (was 2): 2 concurrent trials x MAX_WORKERS=4 threads (up to 8 concurrent
    # in-process graphrag local/global_search calls, each itself multiple LLM/embedding calls)
    # appears to trigger real vLLM-side stalls under this load, not just occasional hangs -
    # observed as every subsequent query timing out for 15+ minutes straight with zero
    # progress despite _run_sync's timeout-recovery working correctly. Serializing trials
    # trades speed for actually finishing overnight.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=callbacks, catch=(openai.APIConnectionError, openai.APITimeoutError, openai.APIError, TimeoutError))

    best = study.best_trial
    if best.value == _REJECTED:
        # Every point in the grid disabled the detector's evidence source. Returning {} leaves the
        # detector on its constructor defaults rather than writing a fabricated best-params entry
        # that later runs would silently reuse.
        logger.error(
            "%s: every trial fell back on more than %.0f%% of documents - no valid configuration "
            "in the search space. Leaving %s unoptimized (defaults) rather than recording one.",
            name, 100 * _MAX_FALLBACK_RATE, name,
        )
        return {}
    result = {
        "params": best.params,
        "threshold": best.user_attrs["threshold"],
        "threshold_edit": best.user_attrs["threshold_edit"],
        "auc": best.value,
        "fallback_rate": best.user_attrs.get("fallback_rate"),
    }
    logger.info("Optimized %s: params=%s auc=%.3f", name, result["params"], result["auc"])
    return result


def optimize_all(
    detector_names: list[str],
    clean_val: list[dict],
    poisoned_val: list[dict],
    *,
    n_trials: int,
    best_params_path: str | Path,
    history_dir: str | Path,
    force: bool = False,
    embedding_connector=None,
    graph_connector=None,
    nli_scorer=None,
    use_dva: bool = False,
) -> dict[str, dict]:
    """Runs optimize_detector_params() for every name in `detector_names` that has
    tunable params, writes the combined result to best_params_path, and one
    per-trial history JSONL file per detector under history_dir. `use_dva` is forwarded
    to every detector's study - see optimize_detector_params for why that matters.

    Unless force=True, first checks best_params_path for a previous run's results and
    reuses whatever's already there per detector, only searching for names missing from
    the cache - so adding a new tunable detector doesn't force every already-tuned one
    to be redone (or, the other way round, get silently skipped just because the cache
    file happens to already exist). Each study is a from-scratch Optuna run (no
    persistent study storage), so anything not reused here is otherwise fully redone
    even if only a later, unrelated step crashed.
    """
    cached = {} if force else load_best_params(best_params_path)
    if cached:
        logger.info("Reusing cached optimize_all results from %s for: %s (pass force=True to redo all).",
                     best_params_path, sorted(set(detector_names) & set(cached)))

    history_dir = Path(history_dir)
    results: dict[str, dict] = dict(cached)
    for name in detector_names:
        if name not in _TUNABLE or name in cached:
            continue
        if use_dva and not _has_dva_strategy(name):
            # No score_dva of its own: every trial would score new_text through score_n, so the
            # whole study reproduces the n-mode result exactly. Skipped rather than tuned - the
            # detector still runs in n mode, it just isn't measured twice. See
            # detector.has_dva_strategy().
            logger.info("Skipping dva-mode study for '%s' - it has no score_dva of its own, so the "
                        "study would reproduce its n-mode result exactly.", name)
            continue
        results[name] = optimize_detector_params(
            name,
            clean_val,
            poisoned_val,
            n_trials=n_trials,
            embedding_connector=embedding_connector,
            graph_connector=graph_connector,
            nli_scorer=nli_scorer,
            history_path=history_dir / f"{name}_trials.jsonl",
            use_dva=use_dva,
        )
        # Saved after every detector, not just at the end - a later detector crashing/hanging
        # (e.g. one that needs a currently-down API) would otherwise discard already-finished
        # results too, forcing a full redo of everything on the next run instead of just picking
        # up where it left off.
        save_best_params(results, best_params_path)
    return results


def save_best_params(results: dict[str, dict], path: str | Path) -> None:
    write_json(results, path)


def load_best_params(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
