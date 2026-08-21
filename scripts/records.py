"""Reading and scoring saved detector record files, with no plotting and no GPU.

benchmark.py charts and tabulates the detector objects it just ran. Fixing one detector therefore
meant either re-running all of them - a full GPU job, and every LLM-based number shifts through
sampling noise - or overwriting good output with a one-detector version. Everything the charts and
the summary table need is already on disk, so both are rebuilt from the record files instead, and
both read them through this module.

Nothing here may import scripts.detector: that pulls in torch, transformers and a live GraphRAG
connector, none of which a chart or a table needs.
"""
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

# How much corpus state a detector reads, which is the axis the thesis compares along - so it is
# also the axis the charts are grouped by and the summary table's section order. detector.AWARENESS
# is None for text-only detectors, "embedding" for vector-store ones, "grag" for graph ones. Rank
# drives x-axis order; label is what the reader sees.
AWARENESS_ORDER: Dict[Any, tuple] = {
    None: (0, "document-level"),
    "embedding": (1, "embedding-aware"),
    "grag": (2, "grag-aware"),
}
AWARENESS_FALLBACK = (3, "other")

# Mirrors detector.AWARENESS. Duplicated rather than imported, for the reason in the module
# docstring - but duplicated exactly once, here, for every offline consumer.
AWARENESS_BY_NAME = {
    "rand": None, "rand1": None, "perplexity": None, "perplexity_filtering": None,
    "llm_as_a_judge": None,
    "embedding_outlier": "embedding", "nli_contradiction": "embedding",
    "gnlic": "grag", "gtopo": "grag", "gtopo2": "grag", "grag_as_a_judge": "grag",
    "grag_contradiction": "grag",
}

# Chart/table nomenclature: the thesis's baselines table, plus gtopo/gtopo2 - this thesis's own
# methods, not in that table, but given the same short-name treatment for consistency across every
# chart and table (matches scripts/latex_tables.py's LATEX_NAMES).
DISPLAY_NAMES = {
    "rand": "Rand", "rand1": "Rand1", "perplexity": "Perp", "llm_as_a_judge": "LLMaaJ",
    "embedding_outlier": "EmbO", "nli_contradiction": "NLIC", "grag_as_a_judge": "GRAGaaJ",
    "gnlic": "GNLIC", "gtopo": "GTopo", "gtopo2": "GTopo2",
}

# Explicit left-to-right order within an awareness tier, for the detectors where alphabetical is
# not the order a reader wants: the chance baselines come first so every real detector is read
# against them, then the cheapest text-only method, then the LLM judge. Anything unlisted keeps
# alphabetical order after these.
PREFERRED_ORDER = ["Rand", "Rand1", "Perp", "LLMaaJ"]


def base_name(name: str) -> str:
    """'grag_as_a_judge(mistral-small-4,v5)' -> 'grag_as_a_judge'."""
    return re.sub(r"\(.*\)$", "", name)


def awareness_of(d) -> tuple:
    """(rank, label) for a detector object - real or RecordedDetector."""
    return AWARENESS_ORDER.get(getattr(type(d), "AWARENESS", None), AWARENESS_FALLBACK)


def tier_of(name: str) -> tuple:
    """(rank, label) straight from a detector's persisted name, for callers holding no object."""
    return AWARENESS_ORDER.get(AWARENESS_BY_NAME.get(base_name(name)), AWARENESS_FALLBACK)


def display_name(d) -> str:
    """Chart/table label: the detector, without the model/variant suffix its record filename carries.

    `d.name` is provenance ("llm_as_a_judge(mistral-small-4,v9)") and stays that way on disk, but on
    an axis it is unreadable and - worse - splits one detector into several bars when n and dva mode
    happen to select different prompt variants. Collapsing to the class name puts those side by side
    where they belong.
    """
    raw = getattr(d, "name", type(d).__name__)
    return DISPLAY_NAMES.get(base_name(raw), base_name(raw))


def display_name_str(raw: str) -> str:
    """display_name, for callers holding only a persisted name/key (e.g. a best_params dict key or
    a trials filename stem) rather than a detector/RecordedDetector object."""
    return DISPLAY_NAMES.get(base_name(raw), base_name(raw))


# Short, human-readable labels for a benchmark run/attack-set, for chart axes and titles - shared so
# every chart calls the same run by the same name. Keys are the various raw spellings that show up
# across records/best_params/trials directories: a bare dataset name, <dataset>_n/<dataset>_dva
# (best_params/trials file stems), and graphrag_under_fire's attack-type suffix (both with and
# without the trailing _n, since it appears both ways across callers).
SET_DISPLAY_NAMES = {
    "wvc": "WVC", "wvc_n": "WVC (regular)", "wvc_dva": "WVC (DVA)",
    "hotpotqa": "HotpotQA", "musique": "MuSiQue",
    "graphrag_under_fire": "GUF",
    "graphrag_under_fire_direct": "GUF (direct)", "graphrag_under_fire_direct_n": "GUF direct",
    "graphrag_under_fire_indirect": "GUF (indirect)", "graphrag_under_fire_indirect_n": "GUF indirect",
    "graphrag_under_fire_enhanced": "GUF (enhanced)", "graphrag_under_fire_enhanced_n": "GUF enhanced",
}


def set_display_name(raw: str) -> str:
    return SET_DISPLAY_NAMES.get(raw, raw)


def ordered_detectors(detectors: List) -> List:
    """Sorted by awareness category, then PREFERRED_ORDER, then name - so every chart reads
    document-level -> embedding-aware -> grag-aware left to right, and the same detector sits in
    the same place across charts."""
    def key(d):
        name = display_name(d)
        rank = PREFERRED_ORDER.index(name) if name in PREFERRED_ORDER else len(PREFERRED_ORDER)
        return (awareness_of(d)[0], rank, name)
    return sorted(detectors, key=key)


def auc(pos_scores: List[float], neg_scores: List[float]) -> float | None:
    """Area under the ROC curve for two score samples, ties averaged. None if a class is missing.

    The one AUC in the offline path - charts, sliding windows and the summary table all come here.
    Rank-based rather than sklearn's roc_auc_score: it equals it mathematically but computes the
    exact rational value instead of integrating trapezoids, and sklearn's last-bit error was enough
    to flip a printed 0.983 to 0.982. Not importing sklearn also keeps this module cheap.
    """
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if not n_pos or not n_neg:
        return None
    scores = list(pos_scores) + list(neg_scores)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):  # average ranks within ties, matching sklearn
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def records_auc(records: List[dict]) -> float | None:
    """AUC over labelled (score, poisoned) records."""
    return auc([float(r["score"]) for r in records if r["poisoned"]],
               [float(r["score"]) for r in records if not r["poisoned"]])


def confusion(records: List[dict], threshold: float) -> tuple:
    """(tp, fp, tn, fn) at `threshold`. A record counts as flagged when its score exceeds it."""
    tp = sum(1 for r in records if r["poisoned"] and r["score"] > threshold)
    fp = sum(1 for r in records if not r["poisoned"] and r["score"] > threshold)
    fn = sum(1 for r in records if r["poisoned"] and r["score"] <= threshold)
    tn = sum(1 for r in records if not r["poisoned"] and r["score"] <= threshold)
    return tp, fp, tn, fn


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Every threshold-dependent metric the charts and the summary table report. AUC is not here:
    it is threshold-free and comes from the scores, so each caller adds it from whichever source it
    has (a live detector's roc_auc, or records_auc over saved records)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # true negative rate / negative recall
    neg_precision = tn / (tn + fn) if (tn + fn) else 0.0  # negative predictive value
    return {
        "n": tp + fp + tn + fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "mcc": mcc,
        "neg_precision": neg_precision,
        "neg_recall": specificity,
        "informedness": recall + specificity - 1.0,
    }


def compute_metrics(d, use_dva: bool = False) -> Dict[str, float]:
    """Metrics for a detector object - a live one from benchmark.py or a RecordedDetector."""
    return {**metrics_from_counts(d.num_tp, d.num_fp, d.num_tn, d.num_fn),
            "auc": d.roc_auc(use_dva=use_dva)}


def metrics_from_records(records: List[dict], threshold: float) -> Dict[str, float]:
    """Metrics straight from saved records, for callers that build no detector object. `auc` is
    None - not 0.0 - when the file holds only one class, so a table can say "n/a" rather than
    print a fabricated number."""
    return {**metrics_from_counts(*confusion(records, threshold)),
            "auc": records_auc(records)}


def hanley_mcneil_se(value: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUC estimate (Hanley & McNeil 1982)."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = value / (2 - value)
    q2 = 2 * value ** 2 / (1 + value)
    var = (value * (1 - value) + (n_pos - 1) * (q1 - value ** 2)
           + (n_neg - 1) * (q2 - value ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


_SHIM_CLASSES: Dict[str, type] = {}


class RecordedDetector:
    """A detector-shaped stand-in built from one saved record file.

    The chart functions only ever ask a detector for its `name`, its class's AWARENESS, the four
    confusion counts and roc_auc() - all derivable from the stored (score, label) pairs. Real
    detectors cannot be rebuilt offline (nli_contradiction loads a DeBERTa model, grag_as_a_judge
    needs a graph connector and a live LLM endpoint), hence the stand-in. AWARENESS is read off the
    *class* by awareness_of, so one subclass is minted per detector name.
    """

    AWARENESS = None

    def __new__(cls, name, *args, **kwargs):
        if cls is RecordedDetector:
            base = base_name(name)
            if base not in _SHIM_CLASSES:
                _SHIM_CLASSES[base] = type(base, (RecordedDetector,),
                                           {"AWARENESS": AWARENESS_BY_NAME.get(base)})
            cls = _SHIM_CLASSES[base]
        return super().__new__(cls)

    def __init__(self, name: str, records: List[dict], threshold: float, coverage: dict = None):
        self.name = name
        self.display = base_name(name)
        self.coverage_info = coverage
        self.records = records
        self.threshold_n = self.threshold_dva = threshold
        self.num_tp, self.num_fp, self.num_tn, self.num_fn = confusion(records, threshold)
        self.num_positive = self.num_tp + self.num_fn
        self.num_negative = self.num_tn + self.num_fp

    # An instance holds exactly one mode's records - the regular shim is only ever asked with
    # use_dva=False and the dva shim only with True - so the flag is accepted and ignored.
    def roc_auc(self, use_dva: bool = False) -> float:
        return records_auc(self.records) or 0.0

    def summarize_positive_only(self, use_dva: bool = False) -> dict:
        tp = sum(1 for r in self.records if r["score"] > self.threshold_n)
        return {"tp": tp, "fn": len(self.records) - tp,
                "recall": tp / len(self.records) if self.records else None}


# Both filename conventions in use (wvc's _insertions/_edits, logicpoison's _n/_dva) store
# regular-mode records under "records" and dva under "records_edit".
MODE_KEYS = (("n", "records", "threshold", "coverage"),
             ("dva", "records_edit", "threshold_edit", "coverage_edit"))


def current_record_files(records_dir: Path, suffix: str = None) -> List[Path]:
    """The record files belonging to the most recent run, newest-per-detector.

    benchmark.py names a file after the detector's `name`, which carries the prompt variant
    ("llm_as_a_judge(mistral-small-4,v3)_direct.json"). A rerun that picks a different variant
    therefore does not overwrite the previous file, it writes a second one beside it - so a
    directory can hold several files for one detector, all but the newest left over from an earlier
    run. Keeping the newest per (detector, attack group) is what "the results of this run" means;
    any other rule (filename order, best AUC) silently mixes runs.

    A detector that only the older run produced is kept: it has no newer counterpart to supersede
    it. That is a real caveat - the row comes from a different run than its neighbours - so callers
    that care report it, which is what `superseded_record_files` is for.
    """
    newest: Dict[tuple, Path] = {}
    for path in sorted(records_dir.glob("*.json")):
        if path.name == "positive_only_summary.json":
            continue
        if suffix and not path.stem.endswith(suffix if suffix.startswith("_") else f"_{suffix}"):
            continue
        # "llm_as_a_judge(mistral-small-4,v3)_direct" -> ("llm_as_a_judge", "direct"): the group is
        # the last underscore part (never itself underscored - insertions/edits/n/dva/direct/
        # indirect/enhanced), and base_name then drops the variant. So the variant is what
        # collapses, while the attack group and the mode stay apart.
        detector, _, group = path.stem.rpartition("_")
        key = (base_name(detector), group)
        if key not in newest or path.stat().st_mtime > newest[key].stat().st_mtime:
            newest[key] = path
    return sorted(newest.values())


def superseded_record_files(records_dir: Path, suffix: str = None) -> List[Path]:
    """The files current_record_files drops - leftovers from an earlier run, for reporting."""
    kept = set(current_record_files(records_dir, suffix))
    return [p for p in sorted(records_dir.glob("*.json"))
            if p.name != "positive_only_summary.json" and p not in kept
            and (not suffix or p.stem.endswith(suffix if suffix.startswith("_") else f"_{suffix}"))]


def iter_record_files(records_dir: Path, suffix: str = None):
    """Yields (detector name, mode, records, threshold, coverage) for every current record file.

    `suffix` selects one group when a directory holds several: graphrag_under_fire writes all three
    attack types side by side as <detector>_direct.json / _indirect.json / _enhanced.json, so
    without it every attack type would land in one chart.

    The match is on the whole trailing name part, not on the raw string: "_indirect" also ends with
    "direct", so a plain endswith let suffix="direct" pull in the indirect files as well - and since
    load_records keys by detector, the later "_indirect" file then silently overwrote the "_direct"
    one and the direct chart showed indirect numbers.
    """
    for path in current_record_files(records_dir, suffix):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = data.get("detector", path.stem)
        for mode, rec_key, thr_key, cov_key in MODE_KEYS:
            records = data.get(rec_key) or []
            if records:
                yield name, mode, records, data.get(thr_key) or 0.0, data.get(cov_key)


def load_records(records_dir: Path, suffix: str = None) -> tuple:
    """Rebuilds (regular-mode, dva-mode) detectors from every record file in `records_dir`."""
    by_mode: Dict[str, Dict[str, RecordedDetector]] = {"n": {}, "dva": {}}
    for name, mode, records, threshold, coverage in iter_record_files(records_dir, suffix):
        det = RecordedDetector(name, records, threshold, coverage)
        by_mode[mode][det.display] = det
    return list(by_mode["n"].values()), list(by_mode["dva"].values())


def load_corpus_rows(clean_path: Path, poisoned_path: Path) -> tuple:
    """Returns (clean rows, poisoned rows), each keyed by every identifier a record might use.

    Records identify a document either by its POSITION in the corpus file (compare_corpus, so wvc
    and LogicPoison) or by the corpus's own "_id" string (graphrag_under_fire), so both are indexed;
    they cannot collide since one key is an int and the other a str.

    The two corpora are indexed independently: insertion attacks make them different lengths, so
    positions do not line up between them.
    """
    def index(path: Path) -> Dict[Any, dict]:
        out: Dict[Any, dict] = {}
        for i, line in enumerate(l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()):
            row = json.loads(line)
            out[i] = row
            if row.get("_id") is not None:
                out[str(row["_id"])] = row
        return out

    return index(clean_path), index(poisoned_path)
