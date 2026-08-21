import json
import logging
import math
import random
import time
from abc import abstractmethod
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm import tqdm


class UnscorableDocument(Exception):
    """Raised by a detector's score_n/score_dva when a document has no valid score to give -
    e.g. perplexity_filtering on empty/near-empty text, where GPT2's loss is undefined rather
    than merely large. Handled like NoGraphEvidence: skip and count, never fabricate a score
    (fabricating one for an edge case can correlate with the label and leak it - see gnlic's
    no-evidence fallback for the same reasoning applied to missing graph context).
    """


def write_json(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


_PROMPTS_PATH = Path(__file__).parent.parent / "prompts.jsonl"
_prompts_cache: dict[tuple[str, str, int], dict] | None = None


def calibrate_detector(d, clean_docs: list, use_dva: bool = False, poisoned_docs: list | None = None) -> None:
    """Calibrates a detector that has a calibrate(), passing the scoring mode when it accepts one.

    gtopo2 must know the mode: in dva mode it scores the changed region, so its null has to be
    built by running that same code path over clean edits. gtopo v1 takes no such argument (and
    that mismatch - dva scored on differences, null built from absolute values - is the KNOWN
    LIMITATION its own docstring records). Signature check rather than try/except TypeError, which
    would silently swallow a genuine TypeError raised inside calibrate() and retry it.

    `poisoned_docs`, likewise, is forwarded only to a calibrate() that accepts it - it orients each
    signal's tail from labels. Pass ONLY held-out positives (the validation split); passing the
    documents being reported on would fit the direction to the test set.

    Lives here rather than in benchmark.py because scripts/optimize.py needs it too, and
    benchmark.py already imports optimize (importing back would be circular).
    """
    import inspect
    params = inspect.signature(d.calibrate).parameters
    kwargs = {}
    if "use_dva" in params:
        kwargs["use_dva"] = use_dva
    if "poisoned_docs" in params and poisoned_docs:
        kwargs["poisoned_docs"] = poisoned_docs
    d.calibrate(clean_docs, **kwargs)


def _load_prompts() -> dict[tuple[str, str, int], dict]:
    # Cached per process - prompts.jsonl never changes at runtime.
    global _prompts_cache
    if _prompts_cache is None:
        _prompts_cache = {}
        with _PROMPTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                _prompts_cache[(row["detector"], row["mode"], row["variant"])] = row
    return _prompts_cache


def get_prompt(detector_name: str, mode: str, variant: int) -> dict:
    """One prompts.jsonl row by (detector, mode, variant). `variant`, not the LLM's `strategy`
    name, is what pairs n and dva mode - see scripts/build_prompts.py."""
    return _load_prompts()[(detector_name, mode, variant)]


def prompt_variant_count(detector_name: str) -> int:
    return len({v for (d, _m, v) in _load_prompts() if d == detector_name})


# ~24k tokens of the judge's 32768 context (wikitext runs ~2.5 chars/token), leaving room for a
# reasoning response. Uncapped, over-long docs got 400'd and dropped - and since clean diffs are
# far longer than poisoned ones, that loss was length-correlated and inflated the judges' AUC.
_JUDGE_MAX_PROMPT_CHARS = 60_000


def _truncate_for_judge(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    import logging
    logging.warning("Judge input truncated: %d -> %d chars (context budget).", len(text), budget)
    return text[:budget] + "\n[... document truncated to fit the judge's context window ...]"


# Base class: insertions score through score_n()/detect_n(), edits through score_dva()/detect_dva()
# (higher = more likely poisoned). Every score is recorded with its true label for post-hoc AUC.
class detector:
    # Which live corpus state score_n() reads, so predecessor clean docs must be ingested before
    # later ones are scored: "embedding" (LanceDB insert), "grag" (reindex), or None.
    AWARENESS: str | None = None

    # True for detectors dominated by a network call (vLLM judge, embedding server) - those score
    # concurrently in detect_batch(). Local GIL-bound models get nothing from threading.
    PARALLELIZABLE: bool = False
    MAX_WORKERS: int = 8  # the stalls once blamed on concurrency were graphrag's global search
    # overflowing --max-model-len and retrying; vLLM reported Waiting=0 throughout.

    def __init__(self, threshold_n: float = 0.5, threshold_dva: float | None = None, optimize_threshold: bool = False):
        self.threshold_n = threshold_n
        self.threshold_dva = threshold_n if threshold_dva is None else threshold_dva
        self.optimize_threshold = optimize_threshold
        self.records_n: list[dict] = []    # normal-mode (detect_n()) score records
        self.records_dva: list[dict] = []  # dva-mode (detect_dva()) score records
        self.reset()

    @abstractmethod
    def score_n(self, doc: dict) -> float:
        # doc is a diff dict; implementations that only need the result read doc["new_text"].
        pass

    # socred a document explicitly with knowledge of its original version (e.g. for edits, not insertions). Defaults to score_n() 
    def score_dva(self, doc: dict) -> float:
        return self.score_n(doc)

    @classmethod
    # returns true, when a detectors posesses a dva (deliberate version access) adapted scorng strategy, 
    def has_dva_strategy(cls) -> bool:
        return cls.score_dva is not detector.score_dva

    @staticmethod
    def _split_score(result) -> tuple[float, dict | None]:
        """score_n/score_dva may return a bare float, or (score, meta) to report per-document
        diagnostics alongside it - e.g. whether a grag-aware detector had to fall back to its
        non-graph equivalent because the graph had no evidence. Returning the diagnostics rather
        than stashing them on self keeps this thread-safe: PARALLELIZABLE detectors score
        concurrently inside detect_batch, so per-document state on the instance would race.
        """
        if isinstance(result, tuple):
            score, meta = result
            return float(score), (meta or None)
        return float(result), None

    def detect_n(self, doc: dict) -> bool:
        return self._split_score(self.score_n(doc))[0] > self.threshold_n

    def detect_dva(self, doc: dict) -> bool:
        return self._split_score(self.score_dva(doc))[0] > self.threshold_dva

    def record(self, poisoned: bool, score: float, use_dva: bool = False, doc_id=None, meta: dict | None = None) -> None:
        target = self.records_dva if use_dva else self.records_n
        entry = {"doc_id": doc_id, "score": score, "poisoned": poisoned}
        if meta:
            entry["meta"] = meta
        target.append(entry)

    # stores the core metrics inside the detector object
    def store(self, poisoned: bool, poison_detected: bool):
        if poisoned:
            self.num_positive += 1
            if poison_detected:
                self.num_tp += 1
            else:
                self.num_fn += 1
        else:
            self.num_negative += 1
            if poison_detected:
                self.num_fp += 1
            else:
                self.num_tn += 1

    def reset(self):
        self.num_positive = 0
        self.num_negative = 0
        self.num_tp = 0
        self.num_fp = 0
        self.num_tn = 0
        self.num_fn = 0
        self.records_n = []
        self.records_dva = []

    def detect_batch(self, clean_docs: list[dict], poisoned_docs: list[dict], use_dva: bool = False, progress_cb=None, meta_fn=None, ingest_fn=None):
        # Scores every doc, records it against its true label, then derives tp/fp/tn/fn from those records. 
        # meta_fn(doc), if given, attaches per-doc metadata so slices (e.g. by length) can be computed post-hoc. 
        # ingest_fn(doc, poisoned), if given, runs right before scoring each
        # doc - e.g. to actually add it to a live corpus/graph first; policy (which docs, if any,
        # get ingested) lives entirely in the caller's closure, not here. Concurrent scoring (via
        # ThreadPoolExecutor) only kicks in when PARALLELIZABLE and ingest_fn is None - ingest_fn
        # mutates shared corpus/graph state doc-by-doc, so those runs must stay strictly ordered.
        import logging
        import openai
        from scripts.graphrag_manager import NoGraphEvidence
        score_fn = self.score_dva if use_dva else self.score_n
        docs_with_labels = [(d, True) for d in poisoned_docs] + [(d, False) for d in clean_docs]
        no_evidence = {"n": 0}  # dict, not int - mutated from ThreadPoolExecutor worker threads

        def score_one(item):
            doc, poisoned = item
            if ingest_fn:
                ingest_fn(doc, poisoned)
            # Wall-clock per scored document, recorded alongside the score. Measured around
            # score_fn only, so it excludes ingest_fn's reindex/insert - that is corpus-maintenance
            # cost, not detection cost. For PARALLELIZABLE detectors this is per-call latency while
            # MAX_WORKERS of them are in flight, not throughput: divide the batch wall time by the
            # document count if throughput is what's wanted.
            t0 = time.perf_counter()
            try:
                s, score_meta = self._split_score(score_fn(doc))
            except NoGraphEvidence:
                # Retrieval failed outright (not merely "the graph knows nothing" - grag-aware
                # detectors now fall back to their non-graph equivalent for that, see
                # _embedding_nli_max). Nothing to score, so skip and count.
                no_evidence["n"] += 1
                return None
            except UnscorableDocument:
                # e.g. perplexity_filtering on empty/near-empty text - no valid score exists,
                # not just an unusual one. Skip rather than record a NaN/fabricated value.
                no_evidence["n"] += 1
                return None
            except (openai.APIConnectionError, openai.APITimeoutError, openai.APIError, TimeoutError) as e:
                # retry_api_call already retried internally and gave up - one bad doc in an
                # unattended multi-hour run shouldn't lose the whole batch, so skip and continue.
                logging.warning(f"[{getattr(self, 'name', type(self).__name__)}] skipping doc {doc.get('record_id')} after persistent API failure: {e}")
                return None
            return doc, poisoned, s, score_meta, (time.perf_counter() - t0) * 1000.0

        def _record(result):
            doc, poisoned, s, score_meta, elapsed_ms = result
            meta = dict(meta_fn(doc)) if meta_fn else {}
            if score_meta:
                meta.update(score_meta)
            meta["elapsed_ms"] = round(elapsed_ms, 3)
            self.record(poisoned=poisoned, score=s, use_dva=use_dva,
                        doc_id=doc.get("record_id"), meta=meta or None)

        if self.PARALLELIZABLE and ingest_fn is None:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
                results = pool.map(score_one, docs_with_labels)
                for result in results:
                    if progress_cb:
                        progress_cb()
                    if result is None:
                        continue
                    _record(result)
        else:
            for item in docs_with_labels:
                result = score_one(item)
                if progress_cb:
                    progress_cb()
                if result is None:
                    continue
                _record(result)
        total = len(docs_with_labels)
        if no_evidence["n"]:
            logging.warning(
                "[%s] %d/%d documents (%.0f%%) skipped after retrieval failure.",
                getattr(self, "name", type(self).__name__), no_evidence["n"], total,
                100 * no_evidence["n"] / total,
            )
        # Fallback rate is the number that matters for a grag-aware detector: it is the share of its
        # score that came from the non-graph equivalent rather than from the graph. Without it the
        # headline AUC can be mostly nli_contradiction/llm_as_a_judge wearing a grag-aware name.
        cov = self.coverage(use_dva=use_dva)
        if cov["n_fallback"]:
            logging.warning(
                "[%s] graph coverage: %d/%d documents (%.0f%%) fell back to the non-graph "
                "equivalent (%d poisoned, %d clean).",
                getattr(self, "name", type(self).__name__), cov["n_fallback"], cov["n_total"],
                100 * cov["n_fallback"] / max(cov["n_total"], 1),
                cov["n_fallback_poisoned"], cov["n_fallback_clean"],
            )
        self.finalize(use_dva=use_dva)

    def coverage(self, use_dva: bool = False) -> dict:
        """How much of this detector's output actually came from its own evidence source.

        `n_fallback` counts records whose meta carries fallback=True. Split by class because a
        fallback rate that correlates with the label is not a coverage gap, it is a leak - the
        fallback path then partly encodes the answer.
        """
        records = self.records_dva if use_dva else self.records_n
        fb = [r for r in records if (r.get("meta") or {}).get("fallback")]
        return {
            "n_total": len(records),
            "n_fallback": len(fb),
            "n_fallback_poisoned": sum(1 for r in fb if r["poisoned"]),
            "n_fallback_clean": sum(1 for r in fb if not r["poisoned"]),
        }

    def finalize(self, use_dva: bool = False) -> None:
        # (Re)derives threshold (if optimize_threshold) and tp/fp/tn/fn from stored records.
        # Safe to call multiple times - always recomputes from scratch.
        records = self.records_dva if use_dva else self.records_n
        if not records:
            return
        if self.optimize_threshold:
            optimal = self._optimal_threshold(records)
            if use_dva:
                self.threshold_dva = optimal
            else:
                self.threshold_n = optimal
        threshold = self.threshold_dva if use_dva else self.threshold_n
        self.num_positive = self.num_negative = self.num_tp = self.num_fp = self.num_tn = self.num_fn = 0
        for r in records:
            self.store(poisoned=r["poisoned"], poison_detected=r["score"] > threshold)

    @staticmethod
    def _optimal_threshold(records: list[dict]) -> float:
        # Sweeps every score in `records` as a candidate cut and returns the one maximising
        # Youden's J (tpr - fpr) - balances both error types and is threshold-scale-agnostic.
        from sklearn.metrics import roc_curve

        labels = [r["poisoned"] for r in records]
        scores = [r["score"] for r in records]
        if len(set(labels)) < 2:
            # can't optimise without both classes present - fall back to the midpoint
            return (min(scores) + max(scores)) / 2 if scores else 0.5

        fpr, tpr, thresholds = roc_curve(labels, scores)
        # roc_curve prepends a synthetic +inf threshold - drop it, or an uninformative
        # classifier could "optimise" to a threshold no real score can ever exceed.
        finite = thresholds != float("inf")
        fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
        if len(thresholds) == 0:
            return (min(scores) + max(scores)) / 2

        j = tpr - fpr
        best_idx = int(j.argmax())
        return float(thresholds[best_idx])

    def roc_auc(self, use_dva: bool = False) -> float:
        # Threshold-independent ROC AUC from the stored (score, label) pairs.
        from sklearn.metrics import roc_auc_score

        records = self.records_dva if use_dva else self.records_n
        labels = [r["poisoned"] for r in records]
        if len(set(labels)) < 2:
            return 0.0
        scores = [r["score"] for r in records]
        return float(roc_auc_score(labels, scores))

    def summarize_positive_only(self, use_dva: bool = False) -> dict:
        # For runs with no negative set (e.g. LogicPoison on musique/hotpotqa): tp/fn/recall are
        # the only metrics that mean anything without a clean class to compare against - reporting
        # specificity/AUC/etc here would just be a fabricated 0.0, not an absence of signal.
        records = self.records_dva if use_dva else self.records_n
        threshold = self.threshold_dva if use_dva else self.threshold_n
        tp = sum(1 for r in records if r["score"] > threshold)
        fn = len(records) - tp
        return {"tp": tp, "fn": fn, "recall": tp / len(records) if records else None}

    def save_to_file(self, path: str | Path) -> None:
        # Dumps every recorded (doc_id, score, true label) pair plus the threshold(s) used,
        # so confusion matrices/AUC/optimal thresholds can be recomputed without rerunning detection.
        # Keys kept as threshold_edit/records_edit (not threshold_dva/records_dva) - this is the
        # persisted file schema, unrelated to the in-code n/dva rename.
        write_json({
            "detector": getattr(self, "name", type(self).__name__),
            "threshold": self.threshold_n,
            "threshold_edit": self.threshold_dva,
            "optimize_threshold": self.optimize_threshold,
            # How many of these scores came from the detector's own evidence source vs its fallback -
            # see coverage(). Persisted so the AUC in any downstream table can be read against the
            # share of documents the graph actually had anything to say about.
            "coverage": self.coverage(use_dva=False),
            "coverage_edit": self.coverage(use_dva=True),
            "records": self.records_n,
            "records_edit": self.records_dva,
        }, path)


class _random_baseline(detector):
    """Chance-level control. Must sit at AUC 0.5 in expectation - it is the line every real
    detector is read against, so a biased one silently rescales the whole comparison.

    Two traps the obvious implementation falls into, both measured:
      * `random.seed(n)` seeds the GLOBAL random module, so merely constructing this detector
        reseeds the stream every other component draws from (and any unrelated random call
        between two scores shifts this detector's own sequence). Use a private Random.
      * Drawing sequentially from one seeded stream makes a score depend on scoring ORDER, and
        detect_batch scores every poisoned doc before every clean one - so the labels get two
        different halves of one fixed sequence. Those halves have different means, and with the
        old seed=123 that was worth a stable AUC of 0.621 (mean 0.591 poisoned vs 0.464 clean),
        reproduced identically across all three graphrag_under_fire attack types. Changing the
        seed only moves the bias, it does not remove it.

    Seeding per document fixes both: the score is a pure function of the document id, identical
    whenever that document is scored and in whatever order, so it cannot correlate with a label
    that happens to be assigned in blocks. Still fully reproducible across runs - Random(str)
    derives its state from a sha512 of the string, so it does not depend on PYTHONHASHSEED.
    """
    SEED = 0

    def __init__(self, optimize_threshold: bool = False):
        super().__init__(threshold_n=0.5, optimize_threshold=optimize_threshold)

    def score_n(self, doc: dict) -> float:
        # record_id is what detect_batch records against; fall back to the text so a doc without
        # one still gets a stable score rather than an order-dependent draw.
        doc_key = doc.get("record_id")
        if doc_key is None:
            doc_key = doc.get("new_text", "")
        return random.Random(f"{self.SEED}:{doc_key}").random()


class rand(_random_baseline):
    SEED = 123


class rand1(_random_baseline):
    SEED = 456


class _llm_judge_base(detector):
    """Shared mechanics for llm_as_a_judge/grag_as_a_judge: client setup, prompt lookup from
    prompts.jsonl (by `variant` - see get_prompt()), and asking+parsing a single score. Subclasses
    supply score_n()/score_dva() (context-free vs context-injected) and, if the system prompt is
    fixed across the whole batch (true for llm_as_a_judge, NOT for grag_as_a_judge - its prompt
    embeds per-document retrieved context), may override detect_batch() for conversation reuse."""

    def __init__(self, prompt_detector_name: str, model: str = "mistral-small-4", variant: int = 0,
                 threshold_n: float = 5, optimize_threshold: bool = False):
        import os
        from openai import OpenAI
        # Score scale here is 0-10 (or 0-1 for probability-output variants) - the LLM's raw
        # judgement, not necessarily [0,1] like other detectors. Thresholds don't need to share a
        # scale across detectors, only be self-consistent per instance (optimize_threshold handles
        # re-deriving the right cut point for whichever scale this variant happens to use).
        super().__init__(threshold_n=threshold_n, optimize_threshold=optimize_threshold)
        self.model = model
        self.variant = variant
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
            # No explicit timeout previously meant the SDK's 10-minute default, plus its own
            # internal max_retries=2 (silently retrying *underneath* retry_api_call) - a single
            # slow-failing call could cost tens of minutes before our own retry logic even saw
            # it. 60s (not the 30s first tried, which turned out to cut off normal-but-slow
            # responses on this shared endpoint - graphrag's own settings.yaml uses 180s for
            # completions) - retry_api_call still owns the retry policy on top.
            timeout=60.0, max_retries=0,
        )
        self._prompt_insertion = get_prompt(prompt_detector_name, "insertion", variant)
        self._prompt_edit = get_prompt(prompt_detector_name, "edit", variant)

    @staticmethod
    def _parse_score(response_text: str, output_format: str = "int_0_10") -> float | None:
        import re
        # Remove thinking/waiting tags some models emit despite instructions not to.
        response_text = re.sub(r'\[THINK\].*?\[/THINK\]', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'\[等待\].*?\[/等待\]', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'Please.*?:', '', response_text)
        response_text = response_text.strip()

        if output_format == "prob_0_1":
            match = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', response_text)
            return float(match.group(1)) if match else None
        if output_format == "int_0_10_cot":
            # Reasoning may mention unrelated numbers before the final verdict - the explicit
            # "SCORE: <n>" marker (required by every int_0_10_cot prompt) is unambiguous; only
            # fall back to a bare last-number search if a model ignores that instruction.
            match = re.search(r'SCORE:\s*([0-9]|10)\b', response_text, re.IGNORECASE)
            if match:
                return float(match.group(1))
            matches = re.findall(r'\b([0-9]|10)\b', response_text)
            if matches:
                return float(matches[-1])
            import logging
            logging.warning(f"Could not parse int_0_10_cot score from response: {response_text[:200]}")
            return None
        # int_0_10 (default): first bare 0-10 integer, not e.g. a year embedded in prose.
        match = re.search(r'\b([0-9]|10)\b', response_text)
        if match:
            return float(match.group(1))
        import logging
        logging.warning(f"Could not parse score from response: {response_text[:100]}")
        return None

    def combined_texts(self, doc: dict) -> str:
        # Each side gets its own slice of the budget rather than letting _ask() cut the joined
        # string at the tail - that would drop new_text (the thing actually being judged) entirely
        # whenever original_text alone already filled the context. 40% each, not a full half, so
        # the remaining ~20% covers the system prompt (grag_as_a_judge's carries retrieved
        # context) and _ask's own cut usually doesn't fire a second time on top of this one.
        half = int(_JUDGE_MAX_PROMPT_CHARS * 0.4)
        return ("the original document is'" + _truncate_for_judge(doc["original_text"], half)
                + " ' The new version is: '" + _truncate_for_judge(doc["new_text"], half) + "'")

    def _ask(self, system_prompt: str, user_content: str, output_format: str) -> float:
        from scripts.helpers import retry_api_call
        # Budget the user content against whatever the system prompt already costs - for
        # grag_as_a_judge that prompt carries per-document retrieved graph context and is itself
        # far from constant. See _truncate_for_judge for why this is a cut rather than a skip.
        user_content = _truncate_for_judge(user_content, max(1000, _JUDGE_MAX_PROMPT_CHARS - len(system_prompt)))
        try:
            response = retry_api_call(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            score = self._parse_score(response.choices[0].message.content.strip(), output_format)
            tqdm.write(f"[{self.name}] {score}")
            # unparseable replies default to 0 ("no evidence of poisoning") rather than
            # silently dropping the document from the benchmark.
            return score if score is not None else 0.0
        except Exception as e:
            import logging
            logging.error(f"Detection failed: {e}")
            raise


class llm_as_a_judge(_llm_judge_base):
    # One call per document via the base class's (now concurrent-capable) detect_batch() - used
    # to keep a hand-rolled growing conversation across docs to amortize the system prompt over
    # fewer round-trips against the old remote endpoint, but that made every call in a chunk wait
    # on every prior one (inherently sequential) and let earlier documents' verdicts bleed into
    # later ones' context. Against a local vLLM server with prefix caching enabled, single-turn
    # calls get the same system-prompt amortization for free while staying independently
    # parallelizable, so the conversation-reuse trick no longer earns its complexity or its cost
    # to correctness.
    PARALLELIZABLE = True

    def __init__(self, model: str = "mistral-small-4", variant: int = 0, optimize_threshold: bool = False):
        super().__init__("llm_as_a_judge", model=model, variant=variant, optimize_threshold=optimize_threshold)
        self.name = f"llm_as_a_judge({model},v{variant})"

    def score_n(self, doc: dict) -> float:
        p = self._prompt_insertion
        return self._ask(p["system_prompt"], doc["new_text"], p["output_format"])

    def score_dva(self, doc: dict) -> float:
        p = self._prompt_edit
        return self._ask(p["system_prompt"], self.combined_texts(doc), p["output_format"])


class perplexity_filtering(detector):
    def __init__(self, model_name: str = "gpt2", abs_threshold: float = 40.0, ratio_threshold: float = 1.05, optimize_threshold: bool = False):
        super().__init__(threshold_n=abs_threshold, threshold_dva=ratio_threshold, optimize_threshold=optimize_threshold)
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.name = f"perplexity({model_name})"
        # See NLIScorer's device comment - GPU 0 is the vLLM slice, already near its memory
        # ceiling; GPU 1 (the embedding server's slice) has far more headroom for this small model.
        self.device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else \
            ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.context_window_size = self.tokenizer.model_max_length
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def _compute_perplexity(self, text: str) -> float:
        import torch
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_window_size,
        ).to(self.device)
        input_ids = encodings["input_ids"]
        # GPT2's loss shifts input_ids by one to predict the next token, so fewer than 2 tokens
        # (empty/near-empty text) leaves nothing to predict and the model returns a NaN loss
        # rather than raising - silently poisoning every downstream ratio/threshold/roc_curve
        # call with NaN instead of failing where the actual problem is.
        if input_ids.shape[-1] < 2:
            raise UnscorableDocument(f"text too short to score ({input_ids.shape[-1]} token(s)): {text!r}")
        with torch.no_grad():
            loss = self.model(input_ids, labels=input_ids).loss
        return float(torch.exp(loss))

    def score_n(self, doc: dict) -> float:
        ppl = self._compute_perplexity(doc["new_text"])
        tqdm.write(f"[perplexity] ppl={ppl:.1f} abs_threshold={self.threshold_n}")
        return ppl

    def score_dva(self, doc: dict) -> float:
        ppl_orig = self._compute_perplexity(doc["original_text"])
        ppl_new = self._compute_perplexity(doc["new_text"])
        ratio = ppl_new / ppl_orig if ppl_orig else 0.0
        tqdm.write(f"[perplexity] orig={ppl_orig:.1f} new={ppl_new:.1f} ratio={ratio:.2f} ratio_threshold={self.threshold_dva}")
        return ratio


class NLIScorer:
    # Shared local NLI model for contradiction scoring between two texts,
    # used by nli_contradiction and grag_contradiction so it's only loaded once.
    _CONTRADICTION_INDEX = 0  # label order for this checkpoint: contradiction, entailment, neutral

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        import torch
        from sentence_transformers import CrossEncoder
        # No CUDA_VISIBLE_DEVICES restriction on the main process (only vLLM/the embedding
        # server's own subprocesses get pinned via env var), so torch's default device picks
        # GPU 0 - the vLLM slice, already running near its --gpu-memory-utilization ceiling with
        # little headroom left, causing an OOM here. GPU 1 (the embedding server's slice) has
        # far more free memory for this small model, and this deployment always has that second
        # GPU when any GPU is present at all.
        
        device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else \
            ("cuda" if torch.cuda.is_available() else "cpu")
        #TODO: Rework into clean if blocks
        self._model = CrossEncoder(model_name, device=device)
        # (premise, hypothesis) -> contradiction probability. The model runs in eval mode, so this
        # is deterministic and safe to memoize. gnlic's Optuna grid is only 12 points (6 alphas x 2
        # methods) yet runs n_trials=30, and alpha is pure post-hoc arithmetic
        # ((1-alpha)*nli + alpha*hub) that changes neither the premise nor the hypothesis - so
        # without this the same document is re-scored on every trial that shares a `method`,
        # turning ~120 genuinely distinct forward passes into ~1800.
        self._prob_cache: dict[tuple[str, str], float] = {}

    def contradiction_prob(self, premise: str, hypothesis: str) -> float:
        key = (premise, hypothesis)
        if key not in self._prob_cache:
            probs = self._model.predict([(premise, hypothesis)], apply_softmax=True)[0]
            self._prob_cache[key] = float(probs[self._CONTRADICTION_INDEX])
        return self._prob_cache[key]

    def contradiction_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        # One batched forward pass for every (premise, hypothesis) pair instead of one
        # model call per pair - grag_contradiction can pull in hundreds of notes for a
        # single document (hops/top_n), and paying tokenization+forward-pass overhead
        # per note individually is what made its Optuna tuning take hours per trial.
        if not pairs:
            return []
        probs = self._model.predict(pairs, apply_softmax=True)
        return [float(p[self._CONTRADICTION_INDEX]) for p in probs]


def _embedding_nli_max(embedding_connector, nli_scorer: NLIScorer, text: str, k: int) -> float:
    """Max NLI contradiction between `text` and its k nearest neighbours in embedding space.

    This is nli_contradiction's whole n-mode strategy, factored out because it is also what every
    grag-aware detector falls back to when the graph has no evidence for a document. Sharing one
    implementation keeps "the non-graph equivalent" literally the same computation rather than a
    lookalike that could drift.
    """
    vector = embedding_connector.embed(text)
    neighbors = embedding_connector.nearest(vector, k)
    if not neighbors:
        return 0.0
    return max(nli_scorer.contradiction_probs([(n["text"], text) for n in neighbors]))


class nli_contradiction(detector):
    # Flags a document if it contradicts its k nearest neighbours in embedding space.
    # score_dva() instead compares directly against the original text - far more reliable
    # for edits, since a clean doc can coincidentally sit near an unrelated contradictory
    # neighbour, but score_n() still needs the neighbour search since insertions have no original.
    AWARENESS = "embedding"
    # score_n()'s neighbor lookup is a real HTTP call to the (local) embedding server; the NLI
    # scoring itself is local GIL-bound compute, but the network round-trip still overlaps fine
    # across threads.
    PARALLELIZABLE = True

    def __init__(self, embedding_connector, nli_scorer: NLIScorer, k_neighbors: int = 5, threshold: float = 0.5, direct_threshold: float = 0.03, optimize_threshold: bool = False):
        super().__init__(threshold_n=threshold, threshold_dva=direct_threshold, optimize_threshold=optimize_threshold)
        self.embedding_connector = embedding_connector
        self.nli_scorer = nli_scorer
        self.k_neighbors = k_neighbors

    def _max_contradiction_with_neighbors(self, text: str) -> float:
        return _embedding_nli_max(self.embedding_connector, self.nli_scorer, text, self.k_neighbors)

    def score_n(self, doc: dict) -> float:
        neighbor_max = self._max_contradiction_with_neighbors(doc["new_text"])
        tqdm.write(f"[nli_contradiction] neighbor_max={neighbor_max:.3f} threshold={self.threshold_n}")
        return neighbor_max

    def score_dva(self, doc: dict) -> float:
        direct = self.nli_scorer.contradiction_prob(doc["original_text"], doc["new_text"])
        tqdm.write(f"[nli_contradiction] direct={direct:.3f} direct_threshold={self.threshold_dva}")
        return direct


def _cosine_distance(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom else 1.0


class embedding_outlier(detector):
    # score_n(): flags a document whose embedding sits unusually far from its corpus neighbours -
    # the only signal available for insertions, which have no prior version.
    # score_dva(): for edits, the corpus-neighbour z-score is a poor proxy for "did this edit
    # change the meaning" (a doc can start far from its neighbours and stay there after a
    # cosmetic edit, or start close and move after a substantive one). Compares the edit's own
    # before/after embeddings directly instead - same "direct version comparison beats neighbour
    # lookup for edits" reasoning as nli_contradiction.score_dva().
    AWARENESS = "embedding"
    PARALLELIZABLE = True  # both modes are real HTTP calls to the (local) embedding server

    def __init__(self, embedding_connector, k: int = 10, z_threshold: float = 1.74, distance_threshold: float = 0.1, optimize_threshold: bool = False):
        super().__init__(threshold_n=z_threshold, threshold_dva=distance_threshold, optimize_threshold=optimize_threshold)
        self.embedding_connector = embedding_connector
        self.k = k

    def score_n(self, doc: dict) -> float:
        vector = self.embedding_connector.embed(doc["new_text"])
        neighbors = self.embedding_connector.nearest(vector, self.k)
        if not neighbors:
            return 0.0
        avg_dist = sum(n["distance"] for n in neighbors) / len(neighbors)
        stats = self.embedding_connector.corpus_distance_stats(self.k)
        z_score = (avg_dist - stats["mean"]) / (stats["std"] or 1e-9)
        tqdm.write(f"[embedding_outlier] avg_dist={avg_dist:.3f} z={z_score:.2f} threshold={self.threshold_n}")
        return z_score

    def score_dva(self, doc: dict) -> float:
        if not doc.get("original_text"):
            # No prior version to compare against (e.g. an edit record with an empty original) -
            # fall back to the neighbour-based signal rather than fabricating a distance.
            return self.score_n(doc)
        old_vector = self.embedding_connector.embed(doc["original_text"])
        new_vector = self.embedding_connector.embed(doc["new_text"])
        version_dist = _cosine_distance(old_vector, new_vector)
        tqdm.write(f"[embedding_outlier] version_dist={version_dist:.3f} threshold={self.threshold_dva}")
        return version_dist




# class grag_contradiction(detector):
#     # Flags a document that contradicts entities/notes connected to it in the GraphRAG graph.
#     AWARENESS = "grag"

#     def __init__(self, graph_connector, nli_scorer: NLIScorer, embedding_connector=None, hops: int = 1,
#                  threshold: float = 0.5, top_n: int = 5, optimize_threshold: bool = False,
#                  k_neighbors: int = 3):
#         super().__init__(threshold_n=threshold, optimize_threshold=optimize_threshold)
#         self.graph_connector = graph_connector
#         self.nli_scorer = nli_scorer
#         # Used only when graph traversal pulls no notes - see _note_scores.
#         self.embedding_connector = embedding_connector
#         self.k_neighbors = k_neighbors
#         self.hops = hops
#         self.top_n = top_n

#     def _fallback_score(self, text: str) -> float:
#         """Non-graph equivalent, for documents the graph pulled no notes for.

#         An empty note pool used to score 0.0 - identical to a document every note agreed with.
#         Pool size tracks document length here, and the poison is far shorter than the clean text,
#         so that fallback leaked the label: on direct, clean note_max median was 0.574 against 0.004
#         for poisoned. Falling back to embedding-neighbour NLI gives those documents a real score.
#         """
#         if self.embedding_connector is None:
#             raise RuntimeError(f"{type(self).__name__} needs an embedding_connector for its no-evidence fallback")
#         return _embedding_nli_max(self.embedding_connector, self.nli_scorer, text, self.k_neighbors)

#     def _note_scores(self, text: str) -> list[float]:
#         entities = self.graph_connector.find_entities_in_text(text)
#         notes = self.graph_connector.related_texts(entities, self.hops, self.top_n)
#         if not notes:
#             return []
#         return self.nli_scorer.contradiction_probs([(note, text) for note in notes])

#     def score_n(self, doc: dict):
#         # Doesn't override score_dva() - the same graph-neighbourhood check on new_text
#         # works for both n and dva mode, so it just falls back to score_n() here.
#         text = doc["new_text"]
#         scores = self._note_scores(text)
#         if not scores:
#             score = self._fallback_score(text)
#             tqdm.write(f"[grag_contradiction] fallback={score:.3f} n=0 threshold={self.threshold_n}")
#             return score, {"fallback": True, "n_notes": 0}
#         note_max = max(scores)
#         tqdm.write(f"[grag_contradiction] note_max={note_max:.3f} note_mean={sum(scores)/len(scores):.3f} "
#                    f"n={len(scores)} threshold={self.threshold_n}")
#         return note_max, {"fallback": False, "n_notes": len(scores)}


class gnlic(detector):
    # Retrieval-then-single-NLI-check, instead of scoring many hand-traversed notes individually
    # and aggregating (max/mean/weighted): asks GraphRAG's own local_search - the same retrieval
    # route a real downstream query would use, which reads community reports too, unlike
    # grag_contradiction's hand-rolled entity/text-unit traversal - what it currently knows
    # relevant to the document, then runs one NLI contradiction check against that context.
    # Blends in a graph-theoretic prior (hub_percentile) since high-centrality entities are more
    # attractive poisoning targets (Nettack-style intuition: perturbing a hub has outsized
    # downstream effect on everything that touches it).
    AWARENESS = "grag"
    PARALLELIZABLE = True  # query_context() is a real LLM/embedding call through the local servers

    def __init__(self, graph_connector, nli_scorer: NLIScorer, embedding_connector=None, alpha: float = 0.3,
                 threshold: float = 0.5, optimize_threshold: bool = False, community_level: int = 2,
                 method: str = "local", k_neighbors: int = 3):
        super().__init__(threshold_n=threshold, optimize_threshold=optimize_threshold)
        self.graph_connector = graph_connector
        self.nli_scorer = nli_scorer
        # Used only when the graph has no evidence for a document - see score_n.
        self.embedding_connector = embedding_connector
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.community_level = community_level
        # NOT tuned any more, fixed by measurement: "global" retrieved for only ~21/120 documents
        # (graphrag scores all 16 community reports at 0 relevance against a document-as-query and
        # returns its canned NO_DATA_ANSWER), against ~120/120 for "local". See optimize.py.
        self.method = method

    def score_n(self, doc: dict):
        text = doc["new_text"]
        context = self.graph_connector.query_context(text, method=self.method, community_level=self.community_level)
        # query_context returns None when the graph had nothing (blank, or graphrag's canned
        # NO_DATA_ANSWER refusal). Scoring that as nli_score = 0.0 - which is what a *fully
        # corroborated* document gets - made "no evidence" and "no contradiction" indistinguishable,
        # and absence correlates with the poisoned class here, so it leaked the label the wrong way.
        # Fall back to the non-graph equivalent instead (nli_contradiction's strategy: NLI against
        # the k nearest neighbours in embedding space), which is a real judgement rather than a
        # fabricated innocent one. The record carries fallback=True so the share of the score that
        # came from the graph stays visible - see detector.coverage().
        fallback = context is None
        if fallback:
            if self.embedding_connector is None:
                raise RuntimeError("gnlic needs an embedding_connector for its no-evidence fallback")
            nli_score = _embedding_nli_max(self.embedding_connector, self.nli_scorer, text, self.k_neighbors)
        else:
            nli_score = self.nli_scorer.contradiction_prob(context, text)
        entities = self.graph_connector.find_entities_in_text(text)
        hub_score = self.graph_connector.hub_percentile(entities)
        final = (1 - self.alpha) * nli_score + self.alpha * hub_score
        tqdm.write(f"[gnlic] nli={nli_score:.3f} hub={hub_score:.3f} final={final:.3f} "
                   f"{'(fallback) ' if fallback else ''}threshold={self.threshold_n}")
        return final, {"fallback": fallback}


class grag_as_a_judge(_llm_judge_base):
    # Same GraphRAG retrieval as gnlic (GraphConnector.query_context), but judged by an LLM prompt
    # instead of NLI: tests whether an LLM judge does better than NLI when given the same
    # graph-retrieved grounding. Unlike llm_as_a_judge, the system prompt embeds per-document
    # retrieved context (the literal "{context}" placeholder in every prompts.jsonl row for this
    # detector), so each doc genuinely needs its own system prompt and reads live graph state,
    # hence AWARENESS="grag". Both query_context() and the judge call itself are real requests
    # to the local servers, so it benefits from the base class's concurrent detect_batch().
    AWARENESS = "grag"
    PARALLELIZABLE = True

    def __init__(self, graph_connector, model: str = "mistral-small-4", variant: int = 0,
                 community_level: int = 2, optimize_threshold: bool = False, method: str = "local"):
        super().__init__("grag_as_a_judge", model=model, variant=variant, optimize_threshold=optimize_threshold)
        self.graph_connector = graph_connector
        self.community_level = community_level
        self.method = method  # see gnlic - "global" has no embedding dependency, unlike local/basic
        self.name = f"grag_as_a_judge({model},v{variant})"

    # Fallback variant is fixed at 0 rather than tuned: the fallback is meant to be a stable
    # reference strategy ("what a context-free judge would have said"), not a second hidden tuning
    # axis whose selection would be confounded with the graph-grounded one. grag_as_a_judge's own
    # `variant` indexes its own prompt set and need not exist in llm_as_a_judge's.
    _FALLBACK_VARIANT = 0

    def _context_for(self, text: str) -> str | None:
        """Retrieved context, or None when the graph has nothing for this document."""
        return self.graph_connector.query_context(text, method=self.method, community_level=self.community_level)

    def _judge(self, mode: str, text_for_context: str, user_content: str):
        """Judge `user_content`, grounded in graph context when there is any.

        With no context, fall back to the plain context-free llm_as_a_judge prompt rather than
        substituting a placeholder into the {context} slot. Injecting graphrag's canned refusal
        there measurably pushed the judge toward "can't determine" (31-42% of scores sat at exactly
        0, vs 0% for llm_as_a_judge), and a hand-written stand-in only trades one prompt artifact
        for another - it also tells the judge every uncorroborated document is suspicious, which on
        this corpus is true of the clean held-out documents too.
        """
        context = self._context_for(text_for_context)
        if context is None:
            p = get_prompt("llm_as_a_judge", mode, self._FALLBACK_VARIANT)
            return self._ask(p["system_prompt"], user_content, p["output_format"]), {"fallback": True}
        p = self._prompt_insertion if mode == "insertion" else self._prompt_edit
        system_prompt = p["system_prompt"].replace("{context}", context)
        return self._ask(system_prompt, user_content, p["output_format"]), {"fallback": False}

    def score_n(self, doc: dict):
        return self._judge("insertion", doc["new_text"], doc["new_text"])

    def score_dva(self, doc: dict):
        return self._judge("edit", doc["new_text"], self.combined_texts(doc))


class gtopo(detector):
    # v2. v1 combined its signals via max(), which let neighborhood_displacement - saturated near
    # 1.0 for clean AND poisoned docs alike (confirmed via 5-fold CV: standalone AUC 0.56, vs.
    # 0.90/0.74 for the two signals it was drowning out) - dominate the pool almost every time.
    # Same failure mode as grag_contradiction's note_max, one level up. Fixed two ways: a quantile
    # (not max) over all co-mentioned entity pairs, so one saturated/outlier pair can't
    # single-handedly decide the score; and two signals instead of four, each calibrated into a
    # p-value against its own empirical distribution on known-clean documents (calibrate() - must
    # be called before scoring; see optimize.py/benchmark.py for the call site), combined via
    # Fisher's method (-2*sum(log p)) rather than a hand-picked linear blend.
    #   inc (incongruity): does this co-mentioned pair cross communities that essentially never
    #     connect in the clean graph (GraphConnector.pair_surprise)?
    #   novlev (novelty x leverage): is this a *new* (non-existing) connection between two
    #     high-centrality entities - exactly what a rational attacker would target?
    # Both signals (pair_surprise's block rates, has_edge, centrality) are read straight off
    # graph_connector's current graph, which is exactly what changes once a predecessor clean doc
    # in the batch gets really ingested - so, like grag_contradiction/gnlic, this needs the real
    # incremental reindex kept in step with the rest of the batch, not a graph frozen at dataset
    # start: AWARENESS="grag".
    AWARENESS = "grag"

    def __init__(self, graph_connector, threshold: float = 0.5, optimize_threshold: bool = False,
                 level: int = 3, q: float = 0.9, max_pairs: int = 400, null: dict | None = None):
        super().__init__(threshold_n=threshold, optimize_threshold=optimize_threshold)
        self.graph_connector = graph_connector
        self.level = level
        self.q = q
        self.max_pairs = max_pairs
        self.null = null or {}  # {signal name: sorted clean-score array}, set by calibrate()
        # False until calibrate() has real clean observations to work with. Callers must check it:
        # scoring uncalibrated yields one identical constant for every document.
        self.calibrated = bool(self.null)

    def _signals(self, text: str) -> dict[str, float]:
        """Per-signal q-th quantile over co-mentioned entity pairs - a quantile, not a max, so one
        pair can't single-handedly saturate the score.

        `density` (graph-known entities per 1000 chars) is a signal in its own right, not just
        bookkeeping. Without it a document too short to yield two graph-known entities produced
        (0.0, 0.0), which reads as *maximally unsurprising* - i.e. maximally innocent - and the
        poison here is much shorter than the clean text (median 77-295 vs 644 chars), so that
        default silently encoded the label. Calibrating density against the same clean null as the
        other signals makes "anomalously few known entities for its length" suspicious on the
        evidence, rather than by a hand-picked constant.
        """
        entities = self.graph_connector.find_entities_in_text(text)
        density = 1000.0 * len(entities) / max(len(text), 1)
        all_pairs = list(combinations(entities, 2))
        if not all_pairs:
            # No pairs to judge: the pair-based signals ABSTAIN (None) rather than claim 0.0.
            # _p() maps None to 0.5, so an abstention adds the same constant to every document's
            # Fisher score and cannot shift the ranking - unlike 0.0, which actively claimed
            # "least surprising". calibrate() likewise skips abstentions rather than baking them
            # into the null.
            return {"inc": None, "novlev": None, "density": density}
        # Seeded uniform sample, not list(...)[:max_pairs]: find_entities_in_text emits matches in
        # graph-node order, so head-truncation always yields pairs built from the same leading few
        # entities. If those are hubs, has_edge is true for nearly all of them, novlev's quantile
        # pins at 0.0 for every document, its null goes degenerate and _p() returns a constant 0.5 -
        # which is exactly what the recorded guf scores show (every one is 1.386 + f(inc) alone).
        pairs = (all_pairs if len(all_pairs) <= self.max_pairs
                 else random.Random(f"gtopo:{len(all_pairs)}:{text[:64]}").sample(all_pairs, self.max_pairs))
        inc = [s for s in (self.graph_connector.pair_surprise(u, v, self.level) for u, v in pairs) if s is not None]
        novlev = [
            0.0 if self.graph_connector.has_edge(u, v)
            else math.sqrt(max(self.graph_connector.centrality(u), 0.0) * max(self.graph_connector.centrality(v), 0.0))
            for u, v in pairs
        ]
        return {
            # pair_surprise returns None where either entity has no community - dropped rather than
            # counted as 0.0 ("least surprising"), which the code has no basis to claim.
            "inc": float(np.quantile(inc, self.q)) if inc else None,
            "novlev": float(np.quantile(novlev, self.q)) if novlev else None,
            "density": density,
        }

    # Which tail of the clean null is suspicious, per signal. inc/novlev are surprising when HIGH
    # (unusual cross-community pair, novel link between hubs). density is surprising when LOW - a
    # document that names far fewer graph-known entities per character than clean text does is the
    # anomaly, and treating it as right-tailed would score exactly backwards.
    _TAILS = {"inc": "right", "novlev": "right", "density": "left"}

    def _p(self, name: str, x: float | None) -> float:
        """Empirical p-value of `x` against the clean null for signal `name`, on the tail given by
        _TAILS. Returns 0.5 (uninformative) when the signal abstained (x is None), when calibrate()
        hasn't run, or when the null is degenerate - a constant contribution to Fisher's method,
        so it shifts every score equally and can't change the ranking."""
        if x is None:
            return 0.5
        ref = self.null.get(name)
        if ref is None or len(ref) == 0 or np.allclose(ref, ref[0]):
            return 0.5
        floor = 1.0 / (len(ref) + 1)
        if self._TAILS.get(name, "right") == "left":
            return max(np.searchsorted(ref, x, side="right") / len(ref), floor)
        return max(1.0 - np.searchsorted(ref, x) / len(ref), floor)

    def _raw(self, text: str) -> dict[str, float | None]:
        return self._signals(text)

    def _fisher(self, r: dict[str, float | None]) -> float:
        return -2.0 * sum(math.log(max(self._p(k, v), 1e-12)) for k, v in r.items())

    def score_n(self, doc: dict):
        r = self._raw(doc["new_text"])
        final = self._fisher(r)
        abstained = [k for k, v in r.items() if v is None]
        shown = " ".join(f"{k}={'--' if v is None else format(v, '.3f')}" for k, v in r.items())
        tqdm.write(f"[gtopo] {shown} score={final:.3f} threshold={self.threshold_n}")
        # fallback here means "had no entity pairs to judge", the gtopo analogue of no graph
        # evidence - reported through the same coverage channel as the other grag-aware detectors.
        return final, {"fallback": bool(abstained), "abstained": abstained}

    def score_dva(self, doc: dict):
        # Contrastive: scores how much MORE incongruous/novel-leveraged the new text is than the
        # original - an entity's context doesn't become suspicious just because a stable,
        # unremarkable document happens to mention it.
        # KNOWN LIMITATION: these are differences, but calibrate() builds its null from absolute
        # values, so _p() compares the two on different scales. Only n-mode is used for
        # graphrag_under_fire; treat any dva-mode gtopo number as unvalidated until the null is
        # rebuilt from clean differences.
        new_r = self._raw(doc["new_text"])
        old_r = self._raw(doc["original_text"])
        r = {k: (None if new_r[k] is None or old_r[k] is None else new_r[k] - old_r[k]) for k in new_r}
        abstained = [k for k, v in r.items() if v is None]
        return self._fisher(r), {"fallback": bool(abstained), "abstained": abstained}

    def calibrate(self, clean_docs: list[dict]) -> None:
        """Builds the null distribution each signal is calibrated against, from documents already
        known to be clean. Must be called before score_n()/score_dva() are meaningful - without
        it, _p() returns an uninformative 0.5 for everything.

        Sets self.calibrated so callers can tell "calibrated against real clean documents" from
        "silently uncalibrated". With an empty null every score collapses to the same constant
        (-2*sum(log 0.5) over the signals), which yields a fabricated recall of 0.0 rather than an
        error - see benchmark.py's logicpoison path, which used to pass [] here.
        """
        acc: dict[str, list[float]] = {}
        for doc in clean_docs:
            for k, v in self._raw(doc["new_text"]).items():
                if v is not None:  # abstentions aren't observations - see _signals()
                    acc.setdefault(k, []).append(v)
        self.null = {k: np.array(sorted(v)) for k, v in acc.items()}
        self.calibrated = bool(self.null)
        if not self.calibrated:
            logging.warning("[gtopo] calibrate() got no usable clean signal values (%d docs) - "
                            "every score would be an identical constant. Detector left uncalibrated.",
                            len(clean_docs))


class gtopo2(detector):
    """gtopo, rebuilt around the signals that measurably carry the class information.

    gtopo v1 is left in place as the baseline; this is a separate detector, not a rewrite, so the
    existing v1 records stay comparable. Its three v1 signals were re-derived offline on wvc
    (gtopo makes no network calls, so its signals recompute from the parquet index alone) and
    measured standalone:

      inc     AUC 0.501 - pinned at exactly 1.0 for 97% of documents in BOTH classes. Cause is in
                pair_surprise(): it percentile-ranks against the rates of community pairs that DO
                have edges, so any never-connecting pair saturates at the top. Replaced by
                `blockodds`, which keeps the degree-corrected log-odds as a magnitude.
      novlev  AUC 0.520 - has_edge() is false for ~100% of sampled pairs in both classes, leaving
                only a pagerank product both classes share. Replaced by `support`, the
                link-prediction (Adamic-Adar) view of whether the surrounding graph corroborates
                the asserted pair at all.
      density AUC 0.652 on wvc - but raw character length alone scores 0.688 on the same split,
                so this was a document-length proxy, not a topology signal. Worse, its correct
                tail is dataset-dependent: on graphrag_under_fire the poison is a short,
                entity-dense fabricated sentence against a 631-char clean corpus, so v1's
                hardcoded left tail scored it backwards - 54% of GUF indirect/enhanced poison
                landed on the innocence floor, AUC 0.228/0.233. DROPPED as a scored signal;
                length and entity count are reported as meta so the confound stays auditable
                instead of silently driving the score.

    What replaces them is localisation. A LogicPoison entity swap is one changed pair among the
    ~400 v1 sampled, and a quantile over 400 pairs cannot see it (v1 on wvc edits: AUC 0.438).
    Scoring the CHANGED REGION against the document's own stable entity context instead measured
    AUC 0.806 (mean hop distance) and 0.690 (Adamic-Adar) on the same documents.
    """

    # Same rationale as gtopo: every signal reads graph_connector's live graph, which is exactly
    # what a predecessor clean doc's real ingest changes.
    AWARENESS = "grag"

    # Which tail is suspicious is NOT hardcoded here. v1's _TAILS did that and it is what produced
    # the inverted GUF numbers - a signal whose sign flips across corpora is scored backwards with
    # no way to notice. calibrate() picks each signal's direction from the clean null itself.
    _SIGNALS = ("dist", "support", "blockodds", "cohesion")

    def __init__(self, graph_connector, threshold: float = 0.95, optimize_threshold: bool = False,
                 level: int = 2, q: float = 0.5, dist_cap: int = 4, max_pairs: int = 400,
                 max_context: int = 25, null: dict | None = None, signals: str = "all"):
        super().__init__(threshold_n=threshold, optimize_threshold=optimize_threshold)
        self.graph_connector = graph_connector
        self.level = level
        self.q = q
        self.dist_cap = dist_cap
        self.max_pairs = max_pairs
        self.max_context = max_context  # context entities compared against per focus entity;
        # capped by degree so one hub-heavy document doesn't cost quadratically more than the rest.
        self.null = null or {}
        self.calibrated = bool(self.null)
        self._alpha: float | None = None      # Egonet Density Power Law exponent, fit in calibrate()
        self._tails: dict[str, str] = {}      # signal -> "left"/"right", learned in calibrate()
        # Which signals drive the score. All four are always COMPUTED and recorded in meta - this
        # only selects what _combine() averages - so a run stays analysable whatever it scores on.
        # Equal weighting means a signal that carries nothing on a corpus still dilutes the ones
        # that do: measured on wvc's 250/250 report split, the four-signal mean scores AUC 0.469 in
        # n mode where blockodds alone scores 0.570, and 0.677 in dva where dist+cohesion scores
        # 0.708 on the same documents. Which subset wins is corpus- and mode-dependent, so it is
        # tuned on the labelled validation split (see scripts/optimize.py) rather than hardcoded
        # here - the same reasoning that made the tail directions learned rather than fixed.
        self.scored_signals = (tuple(self._SIGNALS) if signals == "all"
                               else tuple(s for s in self._SIGNALS if s in signals.split("+")))
        if not self.scored_signals:
            raise ValueError(f"gtopo2: signals={signals!r} selects none of {self._SIGNALS}")
        self.signals_spec = signals

    # --- signal extraction ------------------------------------------------------------------

    def _focus_and_context(self, doc: dict, use_dva: bool) -> tuple[list[str], list[str], dict]:
        """(focus entities, context entities, meta).

        dva mode: focus = entities the edit INTRODUCED, context = entities the edit left in place.
        This is the localisation the whole detector rests on - it asks "is what was just added
        plausible given what this document already talked about", instead of re-describing the
        whole document the way v1 did.

        n mode: an inserted document has no original to diff, so the document's own entity set is
        the focus and each focus entity's graph neighbourhood is the context. The question becomes
        "does the established graph corroborate the relations this document asserts".
        """
        new_entities = self.graph_connector.find_entities_in_text(doc["new_text"])
        if not use_dva or not doc.get("original_text"):
            meta = {"n_focus": len(new_entities), "n_context": len(new_entities),
                    "n_inserted": None, "n_removed": None}
            return new_entities, new_entities, meta
        old_entities = self.graph_connector.find_entities_in_text(doc["original_text"])
        old_set, new_set = set(old_entities), set(new_entities)
        focus = sorted(new_set - old_set)
        context = sorted(new_set & old_set)
        # Reported, never scored. On wvc "did this edit touch any graph entity at all" separates
        # the classes at AUC 0.96 - but only because LogicPoison rewrites entities while the PAN
        # clean edits are typo-level, so it is a property of how the datasets were built, not of
        # poisoning. Scoring it would buy a headline number that no paraphrasing attacker would
        # ever concede. Kept in meta so the evaluation can condition on it honestly.
        meta = {"n_focus": len(focus), "n_context": len(context),
                "n_inserted": len(focus), "n_removed": len(old_set - new_set)}
        return focus, context, meta

    def _signals(self, doc: dict, use_dva: bool) -> tuple[dict[str, float | None], dict]:
        focus, context, meta = self._focus_and_context(doc, use_dva)
        meta["doc_len"] = len(doc["new_text"])
        meta["n_entities"] = len(self.graph_connector.find_entities_in_text(doc["new_text"]))
        signals: dict[str, float | None] = {k: None for k in self._SIGNALS}

        # cohesion needs only the document's own entity set, so it survives an empty focus.
        stats = self.graph_connector.egonet_stats(
            list(dict.fromkeys(focus + context)) if (focus or context) else []
        )
        if stats and self._alpha is not None:
            # OddBall's Egonet Density Power Law: E ~ N^alpha on normal egonets. The residual
            # log(E_observed) - alpha*log(N) is the deviation, negative for a set of entities
            # sparser than the corpus norm - i.e. one whose asserted relations the graph doesn't
            # share. alpha is fit on clean documents in calibrate(), never assumed.
            signals["cohesion"] = math.log(stats["n_edges"] + 1.0) - self._alpha * math.log(stats["n_nodes"])
        meta["egonet"] = stats

        if not focus or not context:
            # No introduced entity, or nothing stable to judge it against: the pair signals
            # ABSTAIN. They are dropped from the combination entirely rather than given a neutral
            # value - see _combine() for why that matters here and did not in v1.
            return signals, meta

        # Highest-degree context entities first: a focus entity's relationship to the document's
        # well-established subjects is what carries the signal, and this bounds the pair count
        # without the seeded sampling v1 needed.
        ctx = sorted(context, key=lambda name: -self.graph_connector.degree_of(name))[:self.max_context]
        pairs = [(f, c) for f in focus for c in ctx][:self.max_pairs]

        # dist: per focus entity, how far the graph places it from the nearest context entity.
        # The strongest measured localised signal (AUC 0.806). Poison introduces an entity the
        # graph does not connect to what the document was already about.
        per_focus_min = []
        for f in focus:
            per_focus_min.append(min(
                (self.graph_connector.hop_distance(f, c, cap=self.dist_cap) for c in ctx),
                default=self.dist_cap + 1,
            ))
        signals["dist"] = float(np.mean(per_focus_min))

        # support: Adamic-Adar corroboration, negated so that (like every other signal here)
        # larger means more anomalous before the tail is learned.
        aa_scores = [self.graph_connector.pair_support(u, v)[0] for u, v in pairs]
        if aa_scores:
            signals["support"] = -float(np.quantile(aa_scores, 1.0 - self.q))

        # blockodds: degree-corrected block log-odds, negated for the same reason - a strongly
        # negative log-odds (communities that connect far less than their degrees predict) becomes
        # a large positive anomaly value.
        odds = [o for o in (self.graph_connector.pair_logodds(u, v, self.level) for u, v in pairs)
                if o is not None]
        if odds:
            signals["blockodds"] = -float(np.quantile(odds, 1.0 - self.q))

        return signals, meta

    # --- calibration ------------------------------------------------------------------------

    @staticmethod
    def _poison_is_higher(pos: list[float], neg: list[float]) -> bool | None:
        """Does the poisoned sample sit above the clean one? (rank comparison, ties half-counted -
        the AUC of the raw signal). None when either sample is too small to say."""
        if len(pos) < 3 or len(neg) < 3:
            return None
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return (wins / (len(pos) * len(neg))) >= 0.5

    def calibrate(self, clean_docs: list[dict], use_dva: bool = False,
                  poisoned_docs: list[dict] | None = None) -> None:
        """Builds the clean null for each signal, the OddBall exponent, and each signal's tail
        direction - all three from the same known-clean documents, none of them hardcoded.

        `use_dva` matters and is the fix for v1's documented KNOWN LIMITATION: v1 scored dva mode
        on DIFFERENCES but built its null from ABSOLUTE values, so _p() compared the two on
        different scales and every dva number it produced was uninterpretable (measured recall:
        1% on hotpotqa and musique). Here the null is built by running the same code path the
        scoring will use, so it is always on the scoring scale.
        """
        # alpha first: the egonet residual depends on it, so it must exist before _signals() runs.
        # Fit log(E+1) = alpha*log(N) by least squares through the origin over clean documents.
        points = []
        for doc in clean_docs:
            stats = self.graph_connector.egonet_stats(
                self.graph_connector.find_entities_in_text(doc["new_text"])
            )
            if stats and stats["n_nodes"] > 1:
                points.append((math.log(stats["n_nodes"]), math.log(stats["n_edges"] + 1.0)))
        if points:
            xs = np.array([x for x, _ in points])
            ys = np.array([y for _, y in points])
            denom = float(np.dot(xs, xs))
            self._alpha = float(np.dot(xs, ys) / denom) if denom > 0 else 1.0
        else:
            self._alpha = 1.0

        acc: dict[str, list[float]] = {}
        per_doc: list[dict[str, float | None]] = []
        for doc in clean_docs:
            signals, _ = self._signals(doc, use_dva)
            per_doc.append(signals)
            for name, value in signals.items():
                if value is not None:  # abstentions aren't observations
                    acc[name] = acc.get(name, []) + [value]

        self.null = {k: np.array(sorted(v)) for k, v in acc.items() if len(v) >= 2}
        # Tail direction, learned. Every signal above is built so that "larger = more anomalous"
        # is the intended reading, but whether that survives on a given corpus is an empirical
        # question - density's did not, and hardcoding it is what made GUF score backwards. A
        # clean null concentrated at the top of its own range means the corpus puts clean
        # documents on the high side, so the LOW tail is the anomalous one there.
        self._tails = {}
        for name, ref in self.null.items():
            mid = (float(ref[0]) + float(ref[-1])) / 2.0
            self._tails[name] = "left" if float(np.median(ref)) > mid else "right"

        # ...but the clean null can only say where clean documents sit in their own range, never
        # which side the POISON lies on, and those are different questions. A signal can be
        # perfectly calibrated on clean data and still point backwards: measured on
        # graphrag_under_fire with the clean-null rule alone, gtopo2 scored AUC 0.386 (indirect)
        # and 0.463 (enhanced) - below chance, i.e. poison ranked as more plausible than clean.
        # That is the same inversion v1's hardcoded _TAILS produced and that learning them was
        # supposed to fix. Orientation is the one thing that cannot be inferred without labels, so
        # when labelled positives are supplied (the held-out validation split - never the split
        # being reported on) each signal's direction is taken from them instead.
        if poisoned_docs:
            poisoned_signals = [self._signals(doc, use_dva)[0] for doc in poisoned_docs]
            for name in self.null:
                pos = [s[name] for s in poisoned_signals if s.get(name) is not None]
                neg = [s[name] for s in per_doc if s.get(name) is not None]
                higher = self._poison_is_higher(pos, neg)
                if higher is not None:
                    self._tails[name] = "right" if higher else "left"

        self.calibrated = bool(self.null)
        if not self.calibrated:
            logging.warning("[gtopo2] calibrate() got no usable clean signal values (%d docs) - "
                            "scores would carry no information. Detector left uncalibrated.",
                            len(clean_docs))
            return

        # Second pass for the combined null: _combine() maps its statistic to a quantile of what
        # clean documents score, which needs the per-signal nulls and tails to already exist -
        # hence a separate pass over the same cached signal values rather than one loop. Mildly
        # self-referential (the clean docs are ranked against a null they are part of), the same
        # trade optimize.py already documents for calibrating on clean_val.
        combined = []
        for signals in per_doc:
            ps = [self._p(k, v) for k, v in signals.items() if k in self.scored_signals]
            live = [p for p in ps if p is not None]
            if live:
                combined.append(float(np.mean([-math.log(max(p, 1e-12)) for p in live])))
        if len(combined) >= 2:
            self.null["_combined"] = np.array(sorted(combined))

    # --- scoring ----------------------------------------------------------------------------

    def _p(self, name: str, x: float | None) -> float | None:
        """Empirical one-sided p-value of `x` against the clean null, on the learned tail.

        Returns None - not 0.5 - for an abstention or a missing/degenerate null. v1 returned 0.5
        and argued it shifted every score equally; that is only true if every document abstains on
        the same signals, and they do not (wvc insertions: 14/200 fallbacks, 11 poisoned vs 3
        clean, so the imbalance tracked the label). _combine() drops Nones instead.
        """
        if x is None:
            return None
        ref = self.null.get(name)
        if ref is None or len(ref) == 0 or np.allclose(ref, ref[0]):
            return None
        floor = 1.0 / (len(ref) + 1)
        if self._tails.get(name, "right") == "left":
            return max(float(np.searchsorted(ref, x, side="right")) / len(ref), floor)
        return max(1.0 - float(np.searchsorted(ref, x)) / len(ref), floor)

    def _combine(self, signals: dict[str, float | None]) -> tuple[float | None, list[str]]:
        """Abstention-invariant combination: the MEAN of -log(p) over signals that actually fired,
        mapped to [0,1] against the calibrated null of that mean.

        v1 summed -2*log(p) over all three signals with 0.5 substituted for abstentions, so a
        document where two signals abstained sat on a different scale than one where none did.
        Averaging over only the live signals keeps documents comparable however many fired, and
        the final quantile mapping gives the threshold a corpus-independent meaning - 0.95 is
        "more anomalous than 95% of clean documents", on any corpus. That is what v1's raw Fisher
        statistic could not offer, and why its wvc-tuned threshold transferred to hotpotqa as 19%
        recall off a 0.921 validation AUC.
        """
        ps = {k: self._p(k, v) for k, v in signals.items() if k in self.scored_signals}
        live = [p for p in ps.values() if p is not None]
        abstained = sorted(k for k, p in ps.items() if p is None)
        if not live:
            return None, abstained
        stat = float(np.mean([-math.log(max(p, 1e-12)) for p in live]))
        ref = self.null.get("_combined")
        if ref is None or len(ref) == 0:
            # Uncalibrated combined null (calibrate() not run, or run before this signal set
            # existed): report the raw statistic squashed into [0,1) rather than a fabricated
            # quantile, so it is still monotone in anomalousness and still thresholdable.
            return 1.0 - math.exp(-stat), abstained
        return float(np.searchsorted(ref, stat)) / len(ref), abstained

    def _score(self, doc: dict, use_dva: bool):
        signals, meta = self._signals(doc, use_dva)
        score, abstained = self._combine(signals)
        meta["abstained"] = abstained
        meta["signals"] = {k: (None if v is None else round(v, 4)) for k, v in signals.items()}
        # fallback = "no signal fired at all", the gtopo2 analogue of no graph evidence, reported
        # through the same coverage channel the other grag-aware detectors use.
        meta["fallback"] = score is None
        if score is None:
            # No usable evidence. NOT 0.0: scores here are quantiles of the clean null, so 0.0
            # claims "less anomalous than every clean document" - a verdict of innocence from a
            # document nothing could be measured on. Absence of evidence is not evidence of
            # innocence, and it is not class-neutral either: on graphrag_under_fire nothing fires
            # for 90/150 poisoned documents against 17/150 clean, so the 0.0 floor pinned most of
            # the positives beneath every negative that produced any evidence at all and dragged
            # indirect to AUC 0.386 - below chance - while its individual signals were separating
            # the classes at 0.83. 0.5 is the median clean document: no evidence, no opinion.
            # This does not change what gets FLAGGED (0.5 is far below any sane threshold, 0.95 by
            # default), only what gets ranked above what. Still reported as fallback below, so
            # coverage stays visible rather than being laundered into a real-looking score.
            return 0.5, meta
        shown = " ".join(f"{k}={'--' if v is None else format(v, '.3f')}" for k, v in signals.items())
        tqdm.write(f"[gtopo2] {shown} score={score:.3f} threshold={self.threshold_n}")
        return score, meta

    def score_n(self, doc: dict):
        return self._score(doc, use_dva=False)

    def score_dva(self, doc: dict):
        return self._score(doc, use_dva=True)
