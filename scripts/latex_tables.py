"""Emits the thesis tables as LaTeX, straight from the saved detector record files.

Same source and same arithmetic as scripts/charts.py and scripts/summarize_records.py (everything
goes through scripts.records), so a table can never disagree with the chart next to it. No GPU, no
servers, no detector objects - just the stored (score, label) pairs.

Three subcommands, one per section that reads better as a table than as a chart:

  categories   5.4.3 - detector categories across every dataset, one column per dataset/metric,
               best value per column in bold.
  setting      5.3.2 / 5.3.3 - all detectors x all metrics for a single attack setting.
  stability    5.5 - the optimizer's chosen hyperparameters per dataset, with a spread column.

  python scripts/latex_tables.py categories
  python scripts/latex_tables.py setting results/detector_records/graphrag_under_fire --filter indirect
  python scripts/latex_tables.py stability

Where a detector has several prompt-variant record files for one setting
("llm_as_a_judge(mistral-small-4)" and "(...,v3)"), the best-AUC variant is reported and its tag is
printed to stderr, so the table states which variant it means rather than depending on which
filename sorted last.
"""
import argparse
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):  # run as a file path rather than `python -m scripts.latex_tables`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.records import (
    DISPLAY_NAMES, PREFERRED_ORDER, base_name, current_record_files, iter_record_files,
    metrics_from_records, superseded_record_files, tier_of,
)

# Printable detector names. The record files carry provenance ("grag_as_a_judge(mistral-small-4,v5)");
# a table column wants the method.
LATEX_NAMES = {
    "rand": "Rand", "rand1": "Rand1", "perplexity": "Perp",
    "llm_as_a_judge": "LLMaaJ",
    "embedding_outlier": "EmbO", "nli_contradiction": "NLIC",
    "gnlic": "GNLIC", "gtopo": "GTopo","gtopo2":"GTopo2", "grag_as_a_judge": "GRAGaaJ",
    "grag_contradiction": "GRAG-Cont",
}

# The chance baselines are reported but never win a column: bolding "random" as best would be an
# artefact of noise, not a result.
BASELINES = {"rand", "rand1"}

# Short column heads for the stability table - the best-params stems are far too wide for a column.
SHORT_DATASET = {
    "wvc_n": "WVC $n$", "wvc_dva": "WVC dva", "hotpotqa": "HotpotQA", "musique": "MuSiQue",
    "graphrag_under_fire_direct_n": "GUF dir.",
    "graphrag_under_fire_indirect_n": "GUF indir.",
    "graphrag_under_fire_enhanced_n": "GUF enh.",
    "synthetic_dataset_n": "Synth $n$", "synthetic_dataset_dva": "Synth dva",
}

# Parameters whose values are identifiers, not quantities. `variant` indexes a prompt in
# prompts.jsonl - v9 is not "three more" than v6 - so averaging it would print a number that means
# nothing. Treated as categorical, like gnlic's "method".
NOMINAL_PARAMS = {"variant", "method"}

# (column header, records dir, suffix, mode, metric). AUC needs both classes, so the LogicPoison
# legs - which carry poisoned documents only - report recall instead.
CATEGORY_COLUMNS = [
    ("WVC $n$", "wvc", None, "n", "auc"),
    ("WVC dva", "wvc", None, "dva", "auc"),
    ("GUF direct", "graphrag_under_fire", "direct", "n", "auc"),
    ("GUF indirect", "graphrag_under_fire", "indirect", "n", "auc"),
    ("GUF enhanced", "graphrag_under_fire", "enhanced", "n", "auc"),
    ("Synth $n$", "synthetic_dataset", None, "n", "auc"),
    ("Synth dva", "synthetic_dataset", None, "dva", "auc"),
    ("HotpotQA $n$", "hotpotqa", None, "n", "recall"),
    ("HotpotQA dva", "hotpotqa", None, "dva", "recall"),
    ("MuSiQue $n$", "musique", None, "n", "recall"),
    ("MuSiQue dva", "musique", None, "dva", "recall"),
]

# Column groups for the categories table's header rule, as (label, number of columns).
CATEGORY_GROUPS = [("AUC (poisoned vs.\\ clean)", 7), ("Recall (positive-only)", 4)]

SETTING_METRICS = [("AUC", "auc"), ("Recall", "recall"), ("Precision", "precision"),
                   ("F1", "f1_score"), ("Informedness", "informedness"), ("MCC", "mcc")]


def latex_name(raw: str) -> str:
    base = base_name(raw)
    return LATEX_NAMES.get(base, DISPLAY_NAMES.get(base, base).replace("_", r"\_"))


def sort_key(raw: str):
    """Tier first, then the chart's within-tier order, so table and chart list detectors alike."""
    base = base_name(raw)
    name = DISPLAY_NAMES.get(base, base)
    rank = PREFERRED_ORDER.index(name) if name in PREFERRED_ORDER else len(PREFERRED_ORDER)
    return (tier_of(raw)[0], rank, name)


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def format_col_header(header: str) -> str:
    """Centers and stacks column headers over right-aligned numeric data."""
    parts = header.split(" ")
    if len(parts) > 1:
        stacked = r"\\".join(parts)
        return f"\\multicolumn{{1}}{{c}}{{\\shortstack{{{stacked}}}}}"
    return f"\\multicolumn{{1}}{{c}}{{{header}}}"


def collect(records_dir: Path, suffix: str = None) -> Dict[str, Dict[str, dict]]:
    """base detector name -> mode -> metrics, for the current run's record files.

    Which file counts as current is decided in scripts.records (newest per detector and attack
    group), so charts and tables drop the same leftovers.
    """
    out: Dict[str, Dict[str, dict]] = {}
    for name, mode, records, threshold, coverage in iter_record_files(records_dir, suffix):
        out.setdefault(base_name(name), {})[mode] = {
            **metrics_from_records(records, threshold), "variant": name, "coverage": coverage}
    return out


def tabular(header_rows: List[str], body_rows: List[str], colspec: str,
            caption: str, label: str, note: str = "", font_size: str = r"\scriptsize",
            tabcolsep: str = "2.5pt") -> str:
    lines = [r"\begin{table}[t]", r"  \centering", f"  {font_size}"]
    if tabcolsep:
        lines.append(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    lines += [f"  \\begin{{tabular}}{{{colspec}}}", r"    \toprule"]
    lines += [f"    {row}" for row in header_rows]
    lines.append(r"    \midrule")
    lines += [f"    {row}" for row in body_rows]
    lines += [r"    \bottomrule", r"  \end{tabular}",
              f"  \\caption{{{caption}}}", f"  \\label{{{label}}}"]
    if note:
        lines.insert(-2, f"  \\par\\medskip\\footnotesize {note}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def tier_rows(names: List[str], n_cols: int, cell) -> List[str]:
    """Body rows with a \\multicolumn tier heading wherever the awareness tier changes."""
    rows, tier = [], None
    for raw in sorted(names, key=sort_key):
        this_tier = tier_of(raw)[1]
        if this_tier != tier:
            if tier is not None:
                rows.append(r"\addlinespace")
            tier = this_tier
            rows.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{this_tier}}}}} \\\\")
        rows.append(cell(raw))
    return rows


def bold_best(values: List[float], eligible: List[bool], digits: int = 3) -> List[str]:
    """Formats a column, bolding its single best value. Baselines are formatted but never bold."""
    cells = [fmt(v, digits) for v in values]
    scored = [(v, i) for i, (v, ok) in enumerate(zip(values, eligible))
              if ok and v is not None and not (isinstance(v, float) and math.isnan(v))]
    if scored:
        best = max(scored)[1]
        cells[best] = f"\\textbf{{{cells[best]}}}"
    return cells


# --------------------------------------------------------------------------------------------
# 5.4.3 - detector category comparison
# --------------------------------------------------------------------------------------------
def cmd_categories(args) -> None:
    per_column: List[Dict[str, dict]] = []
    for _, dataset, suffix, mode, _ in CATEGORY_COLUMNS:
        data = collect(args.records_root / dataset, suffix)
        per_column.append({base: modes[mode] for base, modes in data.items() if mode in modes})

    names = sorted({base for col in per_column for base in col}, key=sort_key)
    columns = []
    for col, (_, _, _, _, metric) in zip(per_column, CATEGORY_COLUMNS):
        values = [col.get(base, {}).get(metric) for base in names]
        columns.append(bold_best(values, [base_name(n) not in BASELINES for n in names]))

    def cell(raw: str) -> str:
        i = names.index(raw)
        return " & ".join([latex_name(raw)] + [c[i] for c in columns]) + r" \\"

    group_header = ["&"]
    for label, width in CATEGORY_GROUPS:
        group_header.append(f"\\multicolumn{{{width}}}{{c}}{{\\textbf{{{label}}}}} &")
    
    cmidrules = []
    start = 2
    for _, width in CATEGORY_GROUPS:
        cmidrules.append(f"\\cmidrule(lr){{{start}-{start + width - 1}}}")
        start += width

    header_rows = [
        " ".join(group_header).rstrip("&") + r" \\",
        " ".join(cmidrules),
        "\\textbf{Detector} & " + " & ".join(format_col_header(h) for h, *_ in CATEGORY_COLUMNS) + r" \\",
    ]
    n_cols = 1 + len(CATEGORY_COLUMNS)
    print(tabular(
        header_rows,
        tier_rows(names, n_cols, cell),
        "l" + "r" * len(CATEGORY_COLUMNS),
        "Detector performance by awareness category. Best value per column in bold; the two chance "
        "baselines are excluded from the comparison. AUC is reported where both a poisoned and a "
        "clean class exist; the LogicPoison legs (HotpotQA, MuSiQue) contain poisoned documents "
        "only and are therefore reported as recall at the tuned threshold. "
        "$n$ = insertion mode, dva = document-vs-answer (edit) mode.",
        "tab:detector-categories",
        r"AUC standard error is $\approx 0.04$ at $n=100+100$ and $\approx 0.05$ at $n=60+60$ "
        r"(Hanley \& McNeil); differences below $\approx 0.10$ are not meaningful.",
        font_size=r"\scriptsize",
        tabcolsep="2.5pt",
    ))
    report_variants(dict(zip((h for h, *_ in CATEGORY_COLUMNS), per_column)))
    print("\n[provenance]", file=sys.stderr)
    report_provenance(args.records_root)


def report_variants(per_column: Dict[str, Dict[str, dict]]) -> None:
    """Prints which prompt variant each LLM-based cell used, to stderr so it stays out of the
    LaTeX. A table that says "LLM-as-a-judge" over numbers from four different prompts needs this
    stated somewhere."""
    print("\n[prompt variants used per cell]", file=sys.stderr)
    for column, entries in per_column.items():
        picked = [m["variant"] for base, m in sorted(entries.items()) if "(" in m["variant"]]
        if picked:
            print(f"  {column:16s} {', '.join(picked)}", file=sys.stderr)


# A record file written this much before the newest one in its group did not come from the same
# benchmark run. Runs take a couple of hours, so the window is generous.
STALE_RUN_SECONDS = 6 * 3600


def report_provenance(records_root: Path) -> None:
    """Warns about record files that are leftovers, and about rows that come from an older run."""
    groups = list(dict.fromkeys((dataset, suffix) for _, dataset, suffix, _, _ in CATEGORY_COLUMNS))
    for dataset, suffix in groups:
        records_dir = records_root / dataset
        dropped = superseded_record_files(records_dir, suffix)
        kept = current_record_files(records_dir, suffix)
        if not kept:
            continue
        newest = max(p.stat().st_mtime for p in kept)
        old = [p for p in kept if newest - p.stat().st_mtime > STALE_RUN_SECONDS]
        tag = f"{dataset}/{suffix}" if suffix else dataset
        for path in dropped:
            print(f"  [superseded] {tag}: ignored {path.name} "
                  f"({datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M})",
                  file=sys.stderr)
        for path in old:
            print(f"  [older run]  {tag}: {path.name} is "
                  f"{(newest - path.stat().st_mtime) / 3600:.0f}h older than the rest of this "
                  f"group - its row is not from the same run", file=sys.stderr)


# --------------------------------------------------------------------------------------------
# 5.3.2 / 5.3.3 - one attack setting, all detectors x all metrics
# --------------------------------------------------------------------------------------------
def cmd_setting(args) -> None:
    data = collect(args.records_dir, args.filter)
    modes = [m for m in ("n", "dva") if any(m in v for v in data.values())]
    names = sorted(data, key=sort_key)
    metrics = [m for m in SETTING_METRICS if not (args.positive_only and m[1] == "auc")]

    columns = []
    for mode in modes:
        for _, key in metrics:
            values = [data[n].get(mode, {}).get(key) for n in names]
            columns.append(bold_best(values, [base_name(n) not in BASELINES for n in names]))

    def cell(raw: str) -> str:
        i = names.index(raw)
        return " & ".join([latex_name(raw)] + [c[i] for c in columns]) + r" \\"

    header_rows = []
    if len(modes) > 1:
        parts, start = [], 2
        for mode in modes:
            parts.append(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{mode} mode}}")
            start += len(metrics)
        header_rows.append("& " + " & ".join(parts) + r" \\")
        start = 2
        header_rows.append(" ".join(
            f"\\cmidrule(lr){{{start + i * len(metrics)}-{start + (i + 1) * len(metrics) - 1}}}"
            for i in range(len(modes))))
    header_rows.append(
        "\\textbf{Detector} & " + " & ".join(format_col_header(h) for _, mode in [(0, m) for m in modes]
                                              for h, _ in metrics) + r" \\")

    setting = args.filter or args.records_dir.name
    n_cols = 1 + len(columns)
    print(tabular(
        header_rows,
        tier_rows(names, n_cols, cell),
        "l" + "r" * len(columns),
        f"All detectors on the {setting} setting, grouped by awareness category. "
        "Best value per column in bold; chance baselines excluded from the comparison. "
        "Threshold-dependent metrics use the threshold tuned on the training split.",
        f"tab:setting-{setting.replace('_', '-')}",
        font_size=r"\scriptsize",
        tabcolsep="2.5pt",
    ))


# --------------------------------------------------------------------------------------------
# 5.5 - parameter stability
# --------------------------------------------------------------------------------------------
def spread(values: List[Any], nominal: bool = False) -> str:
    if nominal:
        seen = [str(v) for v in values]
        if not seen:
            return "--"
        top = max(set(seen), key=seen.count)
        if seen.count(top) == 1:
            return f"all differ ({len(seen)} values)"
        return f"{top} ({seen.count(top)}/{len(seen)})"
    nums = [v for v in values if isinstance(v, (int, float))]
    if nums and len(nums) == len(values):
        if len(nums) == 1:
            return fmt(nums[0], 2)
        mean, sd = statistics.mean(nums), statistics.stdev(nums)
        cv = f" (CV {sd / mean:.2f})" if mean else ""
        return f"{mean:.2f} $\\pm$ {sd:.2f}{cv}"
    labels = [v for v in values if isinstance(v, str)]
    if not labels:
        return "--"
    top = max(set(labels), key=labels.count)
    return f"{top} ({labels.count(top)}/{len(labels)})"


def format_stacked_value(value: str) -> str:
    """Breaks a long '+'-joined param spec (e.g. gtopo2's signals="dist+support+blockodds")
    onto two lines with \\shortstack, so it doesn't overflow the narrow right-aligned column
    it shares with short cells like "--" or "0.70"."""
    parts = value.split("+")
    if len(parts) < 3:
        return value
    mid = (len(parts) + 1) // 2
    top, bottom = "+".join(parts[:mid]), "+".join(parts[mid:])
    return f"\\shortstack{{{top}\\\\+{bottom}}}"


def cmd_stability(args) -> None:
    import json

    datasets = args.datasets or [p.stem for p in sorted(args.best_params_dir.glob("*.json"))]
    per_dataset = {ds: json.loads((args.best_params_dir / f"{ds}.json").read_text(encoding="utf-8"))
                   for ds in datasets}
    
    # Map raw JSON keys to their base_name for robust lookup
    normalized_data = {}
    for ds, detectors_dict in per_dataset.items():
        normalized_data[ds] = {base_name(k): v for k, v in detectors_dict.items()}

    detectors = sorted({base_name(d) for e in per_dataset.values() for d in e}, key=sort_key)

    body: List[str] = []
    for detector in detectors:
        entries = [normalized_data[ds].get(detector) for ds in datasets]
        keys = list(dict.fromkeys(k for e in entries if e for k in e["params"]))
        rows = [(k, [(e or {}).get("params", {}).get(k) for e in entries]) for k in keys]
        rows.append(("AUC", [(e or {}).get("auc") for e in entries]))
        if body:
            body.append(r"\addlinespace")
        for i, (key, values) in enumerate(rows):
            name = f"\\multirow{{{len(rows)}}}{{*}}{{{latex_name(detector)}}}" if i == 0 else ""
            cells = [fmt(v, 3) if key == "AUC" else
                     ("--" if v is None else (fmt(v, 2) if isinstance(v, float)
                                               else format_stacked_value(str(v))))
                     for v in values]
            label = key if key == "AUC" else f"\\texttt{{{key.replace('_', chr(92) + '_')}}}"
            body.append(" & ".join([name, label] + cells + [spread(
                [v for v in values if v is not None], key in NOMINAL_PARAMS)]) + r" \\")

    print(tabular(
        ["\\textbf{Detector} & \\textbf{Parameter} & "
         + " & ".join(format_col_header(SHORT_DATASET.get(d, d.replace("_", r"\_"))) for d in datasets)
         + r" & \multicolumn{1}{c}{\textbf{Spread}} \\"],
        body,
        "ll" + "r" * len(datasets) + "l",
        "Hyperparameter settings selected by the optimizer per dataset, with the AUC reached at "
        "that setting. Spread is mean $\\pm$ sd (and coefficient of variation) for numeric "
        "parameters and the modal choice for categorical ones; a low spread means the setting "
        "transfers across datasets rather than needing per-dataset tuning.",
        "tab:parameter-stability",
        r"Requires \texttt{\textbackslash usepackage\{multirow\}}.",
        font_size=r"\scriptsize",
        tabcolsep="2.5pt",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("categories", help="5.4.3 detector category comparison")
    p.add_argument("--records-root", type=Path, default=Path("results/detector_records"))
    p.set_defaults(func=cmd_categories)

    p = sub.add_parser("setting", help="5.3.2/5.3.3 one setting, all detectors x metrics")
    p.add_argument("records_dir", type=Path)
    p.add_argument("--filter", default=None, help="record-file suffix, e.g. indirect")
    p.add_argument("--positive-only", action="store_true", help="no clean class - drop the AUC column")
    p.set_defaults(func=cmd_setting)

    p = sub.add_parser("stability", help="5.5 parameter stability")
    p.add_argument("--best-params-dir", type=Path, default=Path("results/detector_best_params"))
    p.add_argument("datasets", nargs="*", help="best-params stems, e.g. wvc_n hotpotqa musique")
    p.set_defaults(func=cmd_stability)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()