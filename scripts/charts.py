"""Every chart the benchmark produces, drawn from the record files saved on disk (see
scripts/records.py, which does all the reading and scoring) - never from live detector objects, so
this module needs no GPU, no servers, and no torch import.

main.py calls generate_all_charts(conf) once, right after run_benchmark(conf) - it redraws every
detect/curve/optimization/stability chart from whatever is on disk, using the same config paths
the benchmark run itself used. To redraw by hand (e.g. after editing a chart function, with no
rerun), from a Python shell in the project root:

  from scripts.config_loader import load_config, get_config
  from scripts.charts import generate_all_charts
  load_config("config.yaml")
  generate_all_charts(get_config())
"""
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from scripts.records import (
    RecordedDetector, auc, awareness_of, compute_metrics, display_name, display_name_str,
    hanley_mcneil_se, load_corpus_rows, load_records, ordered_detectors, set_display_name,
)


# --------------------------------------------------------------------------------------------
# Shared plumbing: theme, palette, and the axis furniture every chart repeats
# --------------------------------------------------------------------------------------------

def _prepare(out_dir: Path):
    """(pyplot, seaborn) with the shared theme applied and `out_dir` created, or (None, None).

    Every chart function starts this way, and every one of them treats a missing plotting stack as
    "skip the chart" rather than an error - benchmark runs must not die because the machine has no
    matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return None, None
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    return plt, sns


def _save(fig, out_dir: Path, name: str, kind: str) -> None:
    import matplotlib.pyplot as plt
    out_png = out_dir / name
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Wrote {kind} -> {out_png.name}")


# A dva/edit bar is a tint of its detector's colour, not a grey version of it. Most of the
# saturation is kept (0.35 pushed every hue to the same mud) and the bar is lightened instead, so
# the pair still reads as one detector while the modes stay distinguishable in greyscale.
_DVA_SATURATION = 0.90
_DVA_LIGHTEN = 0.45


# One colour family per awareness tier. Nine evenly spaced hues off a colour wheel gave every
# detector its own colour but read as a rainbow, and spent the strongest visual signal on an
# arbitrary axis. Tiers are what the thesis actually compares, so they get the hue; detectors are
# separated by shade within their tier.
_TIER_COLORMAPS = {0: "Blues", 1: "Greens", 2: "Oranges"}
_SHADE_RANGE = (0.45, 0.85)  # mid-to-dark slice: paler is unreadable, darker loses the tint step


def _detector_colors(ordered: List[RecordedDetector]) -> Dict[str, tuple]:
    """One stable colour per detector: its tier's hue, shaded by position within that tier."""
    import numpy as np
    from matplotlib import colormaps

    by_tier: Dict[int, List[str]] = {}
    for d in ordered:
        by_tier.setdefault(awareness_of(d)[0], []).append(display_name(d))

    colors: Dict[str, tuple] = {}
    for rank, names in by_tier.items():
        cmap = colormaps[_TIER_COLORMAPS.get(rank, "Greys")]
        # A single detector in a tier sits mid-range rather than at either extreme.
        points = ([sum(_SHADE_RANGE) / 2] if len(names) == 1
                  else np.linspace(*_SHADE_RANGE, len(names)))
        for name, point in zip(names, points):
            colors[name] = cmap(float(point))[:3]
    return colors


def _dva_variant(color: tuple) -> tuple:
    import seaborn as sns
    # Produce a clearly desaturated, greyer variant for DVA bars so they read as
    # a muted version of the detector colour (better for grayscale printing).
    desat = sns.desaturate(color, _DVA_SATURATION)
    # Lighten the desaturated colour towards white so the DVA bars stay recognizably
    # the detector's hue but read as a lighter variant (better in print than full grey).
    LIGHTEN_TOWARDS_WHITE = 0.45
    return tuple(desat[i] + (1.0 - desat[i]) * LIGHTEN_TOWARDS_WHITE for i in range(3))


def _recolor_bars(ax, ordered: List[RecordedDetector], dva_containers: set) -> None:
    """Repaints an already-drawn bar axes so colour encodes the detector, not the hue variable.

    seaborn maps colour to `hue` (mode/awareness), which spends the whole palette on two levels and
    leaves every detector looking alike. Drawing first and repainting after is deliberate: it keeps
    seaborn's dodging/ordering logic, which is what positions paired n/dva bars correctly. Each
    BarContainer holds one bar per x category in `order`, so container index -> mode and patch
    index -> detector.
    """
    colors = _detector_colors(ordered)
    order = [display_name(d) for d in ordered]
    for c_idx, container in enumerate(ax.containers):
        is_dva = c_idx in dva_containers
        for patch in container:
            # Identify the detector by the bar's x position, NOT its index in the container: a
            # detector with no dva record contributes no patch to the dva container, so indices
            # stop lining up with `order` the moment any detector is n-only (measured: perplexity's
            # dva bar came out in rand's colour). Dodged bars stay within +/-0.5 of their category
            # centre, so rounding the centre recovers the category.
            idx = round(patch.get_x() + patch.get_width() / 2)
            if not 0 <= idx < len(order):
                continue
            base = colors[order[idx]]
            patch.set_facecolor(_dva_variant(base) if is_dva else base)
            patch.set_edgecolor("0.25" if is_dva else "none")
            patch.set_linewidth(0.4 if is_dva else 0)


# The internal mode names are historical ("insertions"/"edits" in the wvc path, "n"/"dva" in the
# logicpoison one). On a chart they should read as the thing being compared: whether the detector
# got the original document to compare against.
_MODE_LABELS = {"insertions": "regular", "n": "regular",
                "edits": "DVA deliberate version access", "dva": "DVA deliberate version access"}


def _is_dva_mode(label: str) -> bool:
    return label in ("edits", "dva")


def _mode_legend_handles(labels: List[str]):
    """Legend explaining the saturation convention, in neutral grey - the colours themselves now
    mean 'which detector', so a per-colour legend would duplicate the x axis."""
    from matplotlib.patches import Patch
    grey = (0.35, 0.42, 0.55)
    out = []
    for label in labels:
        dva = _is_dva_mode(label)
        out.append(Patch(facecolor=_dva_variant(grey) if dva else grey,
                         edgecolor="0.25" if dva else "none", linewidth=0.4,
                         label=_MODE_LABELS.get(label, label)))
    return out


def _stacked_color_legend(ax, order: List[str], colors: Dict[str, tuple], fontsize: int = 15,
                          swatch: float = 0.032, entry_gap: float = 0.055) -> None:
    """Legend to the right of `ax`: one colour swatch per detector, its name written below it (not
    beside it), with `entry_gap` of blank space between one name and the next swatch.

    A standard matplotlib legend puts colour and label on the same line, which reads fine with two
    or three curves but turns into a wall of small same-height text once a chart has nine or ten
    detectors on it - the swatch-then-name stack below gives each entry more vertical room to be
    picked out, at the cost of the extra height this uses (the caller centres it like the legend it
    replaces).
    """
    from matplotlib.patches import Rectangle

    entry_h = swatch + 0.028 * fontsize / 15 + entry_gap  # swatch + label line + gap to next entry
    n = len(order)
    y0 = 0.5 + entry_h * n / 2  # top of the first swatch, whole block centred like the old legend
    x0 = 1.04
    for i, label in enumerate(order):
        y_top = y0 - i * entry_h
        ax.add_patch(Rectangle((x0, y_top - swatch), swatch * 1.3, swatch, transform=ax.transAxes,
                               facecolor=colors[label], edgecolor="none", clip_on=False))
        ax.text(x0 + swatch * 0.65, y_top - swatch - entry_h * 0.12, label, transform=ax.transAxes,
               ha="center", va="top", fontsize=fontsize, clip_on=False)


def _legend_in_empty_slot(g, n_panels: int, col_wrap: int, handles) -> bool:
    """Puts the legend in the grid's unused bottom-right cells.

    col_wrap leaves (col_wrap - n_panels % col_wrap) empty cells on the last row; parking the
    legend there uses space that is otherwise blank instead of widening the figure. Returns False
    when the grid is exactly full or too small to have a full last row, so the caller can fall back
    to placing it outside.
    """
    axes = g.axes.flatten()
    empty = (col_wrap - n_panels % col_wrap) % col_wrap
    if empty == 0 or len(axes) < col_wrap:
        return False
    last_row = axes[n_panels - 1].get_position()  # bottom-left-most occupied cell
    # Far right of the empty span, vertically centred in the row. Anywhere further left collided
    # with the rotated x tick labels of the panel above, which hang down into this band.
    x = axes[col_wrap - 1].get_position().x0
    y = (last_row.y0 + last_row.y1) / 2
    legend = g.figure.legend(handles=handles, title="mode", loc="center left",
                             bbox_to_anchor=(x, y), bbox_transform=g.figure.transFigure,
                             frameon=False, alignment="left")
    legend._legend_box.align = "left"
    return True


def _awareness_bands(ordered: List[RecordedDetector]) -> List[tuple]:
    """[(label, first_index, last_index), ...] - the x ranges each awareness tier occupies.

    `ordered` must already be ordered_detectors() output (one entry per x tick, tiers contiguous),
    so a tier is just the run of consecutive positions sharing a rank.
    """
    ranks = [awareness_of(d) for d in ordered]
    starts = []
    for i, (rank, label) in enumerate(ranks):
        if not starts or starts[-1][0] != rank:
            starts.append((rank, label, i))
    return [(label, start, (starts[idx + 1][2] - 1 if idx + 1 < len(starts) else len(ranks) - 1))
            for idx, (_rank, label, start) in enumerate(starts)]


def _style_detector_axis(ax, bands: List[tuple], labels: bool = False) -> None:
    """Shared look of every x-axis that lists detectors: vertical tick labels, and dotted dividers
    between awareness tiers so the grouping survives greyscale printing (and, on the bar charts,
    survives `hue` being spent on the mode split instead)."""
    ax.tick_params(axis="x", rotation=90)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    for _label, _start, end in bands[:-1]:
        ax.axvline(end + 0.5, color="0.55", linewidth=0.8, linestyle=":", zorder=0)
    if labels:
        for label, start, end in bands:
            # Sized to the tier's own width, not one fixed size for all three: "embedding-aware"
            # over a 2-detector-wide tier collided with its neighbours at the fixed size that a
            # 4-wide tier like "document-level" has plenty of room for.
            width = end - start + 1
            fontsize = min(10, max(6, 1.5 * width + 3))
            ax.text((start + end) / 2, 1.01, label, ha="center", va="bottom", fontsize=fontsize,
                    color="0.35", transform=ax.get_xaxis_transform())


def _categorical_xticks(ax, x: List[int], labels: List[str]) -> None:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)


# --------------------------------------------------------------------------------------------
# Bar charts
# --------------------------------------------------------------------------------------------

# One panel per metric, in reading order rather than dict order. "n" is deliberately absent: it is
# a count, and a count on a 0-1 axis would be an invisible bar next to nine rates.
BAR_METRICS = ["precision", "recall", "f1_score", "accuracy", "mcc",
               "neg_precision", "neg_recall", "informedness", "auc"]


def _draw_metric_bars(rows: List[dict], ordered: List, out_dir: Path, name: str, kind: str,
                      hue_order: List[str], dva_containers: set, y_label: str,
                      mode_legend: bool, col_wrap: int = 4,
                      height: float = 3.5, aspect: float = 1.25) -> None:
    """The bar chart both public bar functions draw: one panel per metric, detectors on x grouped
    left-to-right by awareness tier, colour encoding the detector and tint the mode.

    `rows` is [{detector, metric, value, mode}, ...]; `ordered` is one entry per x tick.
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt, sns = _prepare(out_dir)
    if sns is None:
        return

    order = list(dict.fromkeys(display_name(d) for d in ordered))
    bands = _awareness_bands(ordered)
    n_panels = df["metric"].nunique()

    g = sns.catplot(
        data=df,
        x="detector",
        y="value",
        col="metric",
        col_wrap=min(col_wrap, n_panels),
        kind="bar",
        order=order,
        hue="mode",
        hue_order=hue_order,
        legend=True,
        height=height,
        aspect=aspect,
    )
    g.set_titles("{col_name}" if n_panels > 1 else "", size=15)
    g.set_axis_labels("Detector", y_label, fontsize=14)
    for ax in g.axes.flatten():
        # Colour encodes the detector; the dva/edit pass gets the desaturated variant of it.
        _recolor_bars(ax, ordered, dva_containers)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="both", labelsize=13)
        # Tier names on the first panel only; repeating them on all nine is noise.
        first = ax is g.axes.flatten()[0]
        if first:
            # Category names go above the axes, so the panel title needs extra headroom or the two
            # overlap - matplotlib places both at y~1.0 by default.
            ax.set_title(ax.get_title(), pad=18)
        _style_detector_axis(ax, bands, labels=first)

    if g.legend is not None:
        g.legend.remove()
    if mode_legend:
        handles = _mode_legend_handles(hue_order)
        # Prefer to park the legend in an unused grid slot; if none, place it above the
        # entire figure (centered) so it doesn't overlap panels or widen the plot.
        if not _legend_in_empty_slot(g, n_panels, col_wrap, handles):
            # Move the legend slightly higher to avoid overlapping the awareness tier labels
            # drawn above the x axis, and omit the title to save vertical space.
            legend = g.figure.legend(handles=handles, title=None, loc="upper center",
                                     bbox_to_anchor=(0.5, 1.08), ncol=len(hue_order), frameon=False,
                                     fontsize=11)
            if legend is not None:
                legend._legend_box.align = "left"

    _save(g.figure, out_dir, name, kind)


def create_bar_chart(detectors: List[RecordedDetector], out_dir: Path = Path("plots"),
                     name: str = "detector_metrics_bar_chart.png", use_edit: bool = False,
                     detectors_edit: List[RecordedDetector] | None = None,
                     metrics: List[str] | None = None) -> None:
    """Per-metric bar chart, detectors grouped left-to-right by awareness category.

    Pass `detectors_edit` to put insertion (n) and edit (dva) bars side by side for the same
    detector in ONE chart instead of emitting two charts that can't be compared at a glance. Only
    detectors present in both sets get an edit bar - the ones without their own score_dva are
    deliberately not run in dva mode (see detector.has_dva_strategy()), so they simply show a
    single insertion bar rather than a duplicate of it.
    """
    grouped = detectors_edit is not None
    rows: List[Dict[str, Any]] = []

    # Resolve which metrics to plot for this chart (defaults to BAR_METRICS).
    metrics_list = metrics or BAR_METRICS

    def _add(objs, mode: str, edit: bool) -> None:
        for d in objs:
            metrics = compute_metrics(d, use_dva=edit)
            for metric_name in metrics_list:
                rows.append({"detector": display_name(d), "metric": metric_name,
                             "value": metrics[metric_name], "mode": mode})

    if grouped:
        _add(detectors, "insertions", False)
        _add(detectors_edit, "edits", True)
        # Dedupe on the DISPLAY name: n and dva mode can select different prompt variants of the
        # same detector, which must share one x position rather than becoming two bars.
        ordered = ordered_detectors(list(detectors) + [
            d for d in detectors_edit if display_name(d) not in {display_name(o) for o in detectors}
        ])
    else:
        _add(detectors, "edits" if use_edit else "insertions", use_edit)
        ordered = ordered_detectors(detectors)

    # metrics_list already set above
    _draw_metric_bars(rows, ordered, out_dir, name, "bar chart",
                      hue_order=["insertions", "edits"] if grouped else ["insertions"],
                      dva_containers={1} if grouped else set(),
                      y_label="Score", mode_legend=grouped)


def create_positive_only_chart(entries: List[tuple], out_dir: Path = Path("plots"),
                               name: str = "positive_only.png") -> None:
    """Recall-only chart for runs with no negative set (LogicPoison on musique/hotpotqa).

    `entries` is [(detector_obj, mode, summary_dict), ...] where summary_dict comes from
    detector.summarize_positive_only(). It is the same bar chart as create_bar_chart with exactly
    one metric: without a clean class, precision/specificity/AUC are undefined, and plotting them
    would render fabricated 0.0s as if they were measurements. Recall (tp / positives) is the only
    honest metric here.
    """
    if not entries:
        return
    rows = [{"detector": display_name(d), "metric": "recall", "value": summary["recall"],
             "mode": mode}
            for d, mode, summary in entries if summary.get("recall") is not None]
    if not rows:
        return

    # Dedupe by name FIRST: entries holds one row per (detector, mode), so the same detector appears
    # twice. Ranking the duplicated list would put the category boundaries at the wrong x positions
    # (the axis has one tick per detector, not per row) and push the last label off the plot.
    unique: Dict[str, Any] = {}
    for d, _mode, _summary in entries:
        unique.setdefault(display_name(d), d)
    ordered = ordered_detectors(list(unique.values()))

    hue_order = [m for m in ("n", "dva") if m in {r["mode"] for r in rows}]
    _draw_metric_bars(rows, ordered, out_dir, name, "positive-only chart",
                      hue_order=hue_order,
                      dva_containers={hue_order.index("dva")} if "dva" in hue_order else set(),
                      y_label="Recall (positives only)", mode_legend=True,
                      # One panel, so it sets the whole figure: keep the width growing with the
                      # detector count the way a single-axes chart did.
                      height=4, aspect=max(6.0, 0.55 * len(ordered) + 2) / 4)


# --------------------------------------------------------------------------------------------
# Optimizer diagnostics: two views of the same n-detectors x 2-panels grid
# --------------------------------------------------------------------------------------------

def _load_trial_rows(trials_file: Path) -> List[Dict[str, Any]]:
    rows = []
    with trials_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _plot_param_series(ax, x: List[Any], param_series: Dict[str, List[float]], marker_size: int = 3) -> None:
    """Plot one line per parameter key against a shared x-axis. A single parameter
    gets ax's own y-axis; two or more each get their own y-scale (twinx per key
    beyond the first) so differing ranges (e.g. hops in [1,2] vs top_n in [2,10])
    stay readable instead of being squashed onto one scale."""
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    keys = list(param_series.keys())

    if len(keys) <= 1:
        for key, color in zip(keys, colors):
            ax.plot(x, param_series[key], marker="o", markersize=marker_size, color=color, label=key)
        ax.set_ylabel(keys[0] if keys else "")
        ax.legend(fontsize=12)
        return

    ax.set_ylabel(keys[0], color=colors[0])
    ax.plot(x, param_series[keys[0]], marker="o", markersize=marker_size, color=colors[0], label=keys[0])
    ax.tick_params(axis="y", labelcolor=colors[0])

    twin = ax.twinx()
    twin.grid(False)
    for key, color in zip(keys[1:], colors[1:]):
        twin.plot(x, param_series[key], marker="s", markersize=marker_size, color=color, label=key)
    twin.set_ylabel(", ".join(keys[1:]), color=colors[1])
    twin.tick_params(axis="y", labelcolor=colors[1])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="best")


def _per_detector_grid(items: List[tuple], out_dir: Path, filename: str, suptitle: str, kind: str,
                       draw_row) -> None:
    """One row per detector, two panels wide: `draw_row(ax_left, ax_right, item)` fills a row.

    Both optimizer charts are this grid - a parameter panel beside a value panel, over a shared
    x axis - so the figure sizing, the suptitle offset that keeps it clear of the top row's titles,
    and the layout/save tail live here once.
    """
    if not items:
        return
    plt, _sns = _prepare(out_dir)
    if plt is None:
        return

    fig, axes = plt.subplots(len(items), 2, figsize=(11, 3.2 * len(items)), squeeze=False)
    for row_idx, item in enumerate(items):
        draw_row(axes[row_idx][0], axes[row_idx][1], item)
    fig.suptitle(suptitle, y=1.0 + 0.01 * len(items))
    fig.tight_layout()
    _save(fig, out_dir, filename, kind)


def create_optimization_sweep_chart(dataset_name: str, trials_dir: Path,
                                    out_dir: Path = Path("plots")) -> None:
    """Per detector: AUC-per-trial (left) and the sampled parameter values (right)
    over the same trial axis, so the two panels line up and show which parameter
    region a spike/drop in AUC came from."""
    items = [(f.name.removesuffix("_trials.jsonl"), rows)
             for f in sorted(trials_dir.glob("*_trials.jsonl"))
             if (rows := _load_trial_rows(f))]

    def draw_row(ax_value, ax_param, item) -> None:
        detector_name, rows = item
        name = display_name_str(detector_name)
        trial = [r["trial"] for r in rows]
        ax_value.plot(trial, [r["value"] for r in rows], marker="o", linewidth=0, alpha=0.6,
                      color="#4c72b0", label="trial AUC")
        ax_value.plot(trial, [r["best_value_so_far"] for r in rows], drawstyle="steps-post",
                      color="#dd8452", label="best so far")
        ax_value.set_title(f"{name} - AUC per trial")
        ax_value.set_xlabel("Trial")
        ax_value.set_ylabel("AUC")
        ax_value.set_ylim(-0.05, 1.05)
        ax_value.legend(fontsize=12)

        _plot_param_series(ax_param, trial,
                           {key: [r["params"][key] for r in rows] for key in rows[0]["params"]})
        ax_param.set_title(f"{name} - sampled parameters")
        ax_param.set_xlabel("Trial")

    _per_detector_grid(items, out_dir, f"optimization_sweep_{dataset_name}.png",
                       f"Hyperparameter optimization - {set_display_name(dataset_name)}",
                       "optimization sweep chart", draw_row)


def create_all_optimization_sweep_charts(
    best_params_dir: Path = Path("results/detector_best_params"),
    out_dir: Path = Path("results/charts"),
) -> None:
    for trials_dir in sorted(best_params_dir.glob("*_trials")):
        dataset_name = trials_dir.name.removesuffix("_trials")
        create_optimization_sweep_chart(dataset_name, trials_dir, out_dir)


def _load_best_params_per_dataset(best_params_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """dataset -> detector -> {"params": {...}, "auc": float}.
    Prefers the final <dataset>.json (fit on the full training split); for datasets
    where that file wasn't written (e.g. still mid-run), falls back to the last
    trial's best_params_so_far/best_value_so_far from <dataset>_trials/."""
    per_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for trials_dir in sorted(best_params_dir.glob("*_trials")):
        dataset_name = trials_dir.name.removesuffix("_trials")
        final_json = best_params_dir / f"{dataset_name}.json"
        if final_json.is_file():
            per_dataset[dataset_name] = json.loads(final_json.read_text(encoding="utf-8"))
            continue

        fallback: Dict[str, Dict[str, Any]] = {}
        for trials_file in sorted(trials_dir.glob("*_trials.jsonl")):
            rows = _load_trial_rows(trials_file)
            if rows:
                fallback[trials_file.name.removesuffix("_trials.jsonl")] = {
                    "params": rows[-1]["best_params_so_far"], "auc": rows[-1]["best_value_so_far"]}
        per_dataset[dataset_name] = fallback

    return per_dataset


def _stability_label(key: str, vals: list) -> str:
    """CV for numeric params; for categorical ones (gnlic/grag_as_a_judge's `method`,
    "local"/"global") a mean is meaningless, so report how often the modal choice won instead -
    the categorical analogue of "did the optimizer land in the same place across datasets"."""
    present = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if len(present) == len(vals) and present:
        mean = sum(present) / len(present)
        return f"{key} CV={statistics.stdev(present) / mean:.2f}" if mean and len(present) > 1 else key
    labels = [v for v in vals if isinstance(v, str)]
    if labels:
        top = max(set(labels), key=labels.count)
        return f"{key} {top} {labels.count(top)}/{len(labels)}"
    return key


def _dataset_records_lookup(dataset_name: str) -> tuple:
    """(records_subdir, suffix, mode) for looking up `dataset_name`'s actual detector_records -
    the inverse of how benchmark.py names best_params files (<dataset>_n.json / <dataset>_dva.json,
    with graphrag_under_fire's three attacks folded into the dataset name as a suffix)."""
    if dataset_name.endswith("_n"):
        mode, base = "n", dataset_name[:-2]
    elif dataset_name.endswith("_dva"):
        mode, base = "dva", dataset_name[:-4]
    else:
        return None
    if base.startswith("graphrag_under_fire_"):
        return "graphrag_under_fire", base[len("graphrag_under_fire_"):], mode
    return base, None, mode


def _records_auc_fallback(dataset_name: str, detector_name: str,
                          records_dir: Path) -> float | None:
    """The AUC `detector_name` actually reached on `dataset_name`, read straight from its saved
    records - for detectors like gtopo on graphrag_under_fire, whose Optuna study never found a
    parameter setting that avoided 100% fallback (every trial scored -1, so the optimizer wrote no
    best_params entry at all). The full benchmark run still scored it at its default parameters and
    got a real, non-degenerate AUC (fallback scores still separate the classes somewhat) - that
    number belongs on this chart even though it names no "best setting"."""
    located = _dataset_records_lookup(dataset_name)
    if located is None:
        return None
    subdir, suffix, mode = located
    records_path = records_dir / subdir
    if not records_path.is_dir():
        return None
    dets_n, dets_dva = load_records(records_path, suffix)
    dets = dets_dva if mode == "dva" else dets_n
    match = next((d for d in dets if d.display == detector_name), None)
    return match.roc_auc() if match else None


def create_parameter_stability_chart(
    best_params_dir: Path = Path("results/detector_best_params"),
    out_dir: Path = Path("results/charts"),
    records_dir: Path = Path("results/detector_records"),
) -> None:
    """Per detector: does its best parameter setting stay put across datasets, or
    does the optimizer land somewhere different each time? Left panel plots each
    parameter's chosen value per dataset; right panel plots the AUC that setting
    reached, so a jumpy parameter can be read against whether it actually cost AUC.

    A dataset where the optimizer never landed on a usable setting (empty best_params entry) still
    gets its AUC bar, coloured differently, by falling back to the plain benchmark run's records -
    see _records_auc_fallback. Its parameter panel stays blank: there is no "best setting" to plot.
    """
    per_dataset = _load_best_params_per_dataset(best_params_dir)
    dataset_names = list(per_dataset)
    dataset_labels = [set_display_name(ds) for ds in dataset_names]
    detector_names = sorted({det for entries in per_dataset.values() for det in entries})
    x = list(range(len(dataset_names)))

    def draw_row(ax_param, ax_auc, detector_name) -> None:
        name = display_name_str(detector_name)
        entries = [per_dataset[ds].get(detector_name) for ds in dataset_names]
        # Union of the keys, not the first entry's: a detector can be tuned over different
        # parameters on different datasets (gnlic gained `method` partway through), and indexing
        # one dataset's params with another's key raised KeyError.
        param_keys = list(dict.fromkeys(k for e in entries if e for k in e["params"]))
        param_series = {key: [(e or {}).get("params", {}).get(key, math.nan) for e in entries]
                        for key in param_keys}

        # Categorical params (e.g. gtopo2's `signals`, a "dist+support+blockodds"-style string) get
        # plotted as fake numeric lines on the twin axis, whose long string tick labels blow out the
        # whole column's width for a value that's already summarised in cv_text below - so only
        # numeric params go on the plot itself.
        numeric_param_series = {key: vals for key, vals in param_series.items()
                                if not any(isinstance(v, str) for v in vals)}
        _plot_param_series(ax_param, x, numeric_param_series, marker_size=6)
        cv_text = ", ".join(_stability_label(key, vals) for key, vals in param_series.items())
        ax_param.set_title(f"{name} - best parameter by dataset\n{cv_text}", fontsize=13)
        _categorical_xticks(ax_param, x, dataset_labels)

        aucs = [e["auc"] if e else math.nan for e, ds in zip(entries, dataset_names)]
        fallback = [math.isnan(a) for a in aucs]
        for i, (ds, is_fallback) in enumerate(zip(dataset_names, fallback)):
            if is_fallback:
                aucs[i] = _records_auc_fallback(ds, detector_name, records_dir) or math.nan
        colors = ["#c44e52" if fb and not math.isnan(a) else "#4c72b0"
                 for a, fb in zip(aucs, fallback)]
        ax_auc.bar(x, aucs, color=colors)
        if any(fb and not math.isnan(a) for a, fb in zip(aucs, fallback)):
            from matplotlib.patches import Patch
            ax_auc.legend(handles=[Patch(facecolor="#4c72b0", label="optimizer's best setting"),
                                   Patch(facecolor="#c44e52", label="default params (no valid trial)")],
                          fontsize=9, loc="lower right")
        ax_auc.set_title(f"{name} - AUC at best setting")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_ylim(0, 1.05)
        _categorical_xticks(ax_auc, x, dataset_labels)

    _per_detector_grid(detector_names, out_dir, "parameter_stability_across_datasets.png",
                       "Parameter stability across datasets", "parameter stability chart", draw_row)


# --------------------------------------------------------------------------------------------
# Sliding-window AUC against a per-document property
# --------------------------------------------------------------------------------------------

def _sliding_auc(points: List[tuple], window: int, n_steps: int = 40,
                 neg_pool: List[float] = None):
    """AUC within a sliding window over some per-document quantity.

    `points` is [(x, score, poisoned), ...]. Returns (x, auc, se) lists, one entry per window
    position. Windows overlap and are skipped unless they hold at least 5 of each class.

    `neg_pool` is for axes that only exist for poisoned documents (the changed-token fraction is
    undefined for a document nothing changed in): `points` then holds positives only and every
    window is scored against that one shared list of clean scores instead of against the negatives
    that happen to fall inside the window.
    """
    points = sorted(points, key=lambda t: t[0])
    if len(points) < window:
        return [], [], []
    xs, aucs, ses = [], [], []
    starts = sorted({round(i * (len(points) - window) / max(n_steps - 1, 1)) for i in range(n_steps)})
    for start in starts:
        chunk = points[start:start + window]
        pos = [s for _, s, p in chunk if p]
        neg = neg_pool if neg_pool is not None else [s for _, s, p in chunk if not p]
        if len(pos) < 5 or len(neg) < 5:
            continue
        value = auc(pos, neg)
        xs.append(chunk[len(chunk) // 2][0])
        aucs.append(value)
        ses.append(hanley_mcneil_se(value, len(pos), len(neg)))
    return xs, aucs, ses


def _curve_xlim(drawn_x: List[float], mark_x: List[float], log_x: bool):
    """(lo, hi) covering the stretch the curves actually occupy, plus any reference mark, or None.

    A window spanning `window_frac` of the documents cannot have its centre in the outermost
    documents, so the curves always stop short of the data range - and on a skewed variable
    (first-change: median 0.002, max 0.36) the whole informative part then sits squeezed against one
    edge while most of the axis shows empty space.
    """
    if not drawn_x or min(drawn_x) >= max(drawn_x):
        return None
    lo, hi = min(drawn_x + mark_x), max(drawn_x + mark_x)
    if log_x and lo > 0:
        pad = (hi / lo) ** 0.03
        return lo / pad, hi * pad
    pad = (hi - lo) * 0.03
    return lo - pad, hi + pad


def create_sliding_auc_chart(series: Dict[str, List[tuple]], ordered: List, out_dir: Path,
                             name: str, curve: "_Curve", window_frac: float = 0.4,
                             marks: List[tuple] = None,
                             neg_pool: Dict[str, List[float]] = None,
                             full_range: bool = False) -> None:
    """How each detector's separation of poisoned from clean varies along `curve`'s x axis.

    `series` maps display name -> [(x, score, poisoned), ...]. Each line is a sliding-window AUC, so
    a point at x reads "AUC over documents of about this x" - detection quality with that document
    property largely held fixed. The dotted line at 0.5 is chance; the shaded band is +/-1 standard
    error.

    `neg_pool` (display name -> clean scores) is required by positives-only curves; see
    _sliding_auc. `marks` is [(label, x), ...], drawn as vertical reference lines - e.g. the length
    at which a detector's model starts truncating - in the units of the axis. `full_range` keeps the
    x axis on the whole data range instead of cropping it to the curves; see _fit_x_to_curves.
    """
    if not series:
        return
    plt, _sns = _prepare(out_dir)
    if plt is None:
        return

    colors = _detector_colors(ordered)
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, height_ratios=[4, 1], constrained_layout=True)

    all_x: List[float] = []
    drawn_x: List[float] = []  # only where a curve exists, for the axis limits below
    drawn_labels: List[str] = []  # only detectors that actually got a line, for the legend
    for display in [display_name(d) for d in ordered]:
        points = series.get(display)
        if not points:
            continue
        all_x = [x for x, _, _ in points]  # near-identical across detectors; last one wins
        window = max(int(len(points) * window_frac), 20)
        xs, aucs, ses = _sliding_auc(points, window,
                                     neg_pool=(neg_pool or {}).get(display) if curve.positives_only else None)
        if not xs:
            continue
        drawn_x += xs
        drawn_labels.append(display)
        color = colors[display]
        ax.plot(xs, aucs, color=color, linewidth=1.8, label=display, zorder=3)
        ax.fill_between(xs, [a - e for a, e in zip(aucs, ses)], [a + e for a, e in zip(aucs, ses)],
                        color=color, alpha=0.12, linewidth=0, zorder=2)

    ax.axhline(0.5, color="0.45", linestyle=":", linewidth=1.0, zorder=1)
    # Drop marks the corpus never reaches (e.g. a 512-token NLI cutoff on graphrag_under_fire,
    # whose documents top out under 2000 characters): keeping one would stretch the axis to make
    # room for a reference line with nothing near it, squeezing the actual curve into a sliver.
    marks = [(l, x) for l, x in (marks or []) if all_x and x <= max(all_x) * 1.05]
    for label, x_mark in marks:
        for axis in (ax, ax_hist):
            axis.axvline(x_mark, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
        ax.annotate(label, xy=(x_mark, 0.02), xycoords=("data", "axes fraction"),
                    rotation=90, ha="right", va="bottom", fontsize=13, color="0.35")
    ax.set_ylabel(f"AUC within {curve.window_noun} window", fontsize=20)
    ax.set_ylim(0, 1.02)
    if curve.log_x:
        ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=19)
    _stacked_color_legend(ax, drawn_labels, colors, fontsize=17)
    xlim = None if full_range else _curve_xlim(drawn_x, [x for _label, x in marks], curve.log_x)
    if xlim:
        ax.set_xlim(*xlim)

    # Where the documents actually are, so sparsely covered stretches of the curve are visible.
    # Binned across the visible range rather than the full data: with the axis cropped, full-range
    # bins can be wider than the whole view and collapse the histogram into one block.
    if all_x:
        import numpy as np
        lo, hi = xlim or (min(all_x), max(all_x))
        lo, hi = max(lo, min(all_x)), min(hi, max(all_x))
        if lo < hi:
            bins = (np.logspace(math.log10(lo), math.log10(hi), 30) if curve.log_x and lo > 0
                    else np.linspace(lo, hi, 30))
            ax_hist.hist(all_x, color="0.6", bins=bins)
    ax_hist.tick_params(axis="both", labelsize=19)
    ax_hist.set_ylabel("docs", fontsize=20)
    ax_hist.set_xlabel(curve.x_label, fontsize=20)

    _save(fig, out_dir, name, f"{curve.kind} chart")


# --------------------------------------------------------------------------------------------
# The x axes the sliding-AUC chart can be drawn against. Each is a per-document number derived from
# the (clean row, poisoned row, record) triple, so adding one means adding a row to CURVES - not
# another chart function.
# --------------------------------------------------------------------------------------------

class _Curve:
    def __init__(self, kind: str, x_of, x_label: str, window_noun: str,
                 log_x: bool = False, positives_only: bool = False):
        self.kind = kind
        # x_of(clean_row, poisoned_row, record) -> float, or None to drop the document. The rows are
        # whole corpus rows (text, title, _id, ...) and may be None when a document exists in only
        # one corpus; `record` is the detector's own {doc_id, score, poisoned, meta} entry.
        self.x_of = x_of
        self.x_label = x_label
        self.window_noun = window_noun      # goes into the y label: "AUC within <noun> window"
        self.log_x = log_x
        self.positives_only = positives_only  # x undefined for clean docs; they become the neg pool


def _text(row) -> str:
    return (row or {}).get("text", "")


def _tokens(text: str) -> List[str]:
    """Whitespace tokens. Deliberately not a model tokenizer: the axis should mean the same thing
    for every detector on the chart, several of which use different tokenizers or none at all."""
    return text.split()


def _length_x(clean, poisoned, record):
    # The text this detector actually judged: the poisoned version for a positive, the original for
    # a negative. Only that one has to be present.
    return len(_text(poisoned if record["poisoned"] else clean)) or None


def _diff(clean, poisoned):
    """(clean tokens, poisoned tokens, non-equal opcodes) for one document pair, or None.

    Every axis below except `length` is a different reading of the same token diff, so they all
    come through here. None means the pair cannot be diffed at all: a document the attack
    fabricated outright (insertion attacks have no clean counterpart) or an empty side. Those drop
    out of the chart rather than being charted as a spurious maximum.
    """
    from difflib import SequenceMatcher
    old, new = _tokens(_text(clean)), _tokens(_text(poisoned))
    if not old or not new:
        return None
    opcodes = [op for op in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
               if op[0] != "equal"]
    return old, new, opcodes


def _changed_token_count(opcodes) -> int:
    """A replace counts as many tokens as its longer side, so a swap of one entity for three words
    counts three."""
    return sum(max(i2 - i1, j2 - j1) for _tag, i1, i2, j1, j2 in opcodes)


def _change_fraction_x(clean, poisoned, record):
    """Share of the document the attack rewrote: changed tokens / total tokens."""
    if (d := _diff(clean, poisoned)) is None:
        return None
    _old, new, opcodes = d
    # Deletion-heavy edits can change more tokens than the surviving document has; cap at 1 so the
    # axis stays a share of a document.
    return min(_changed_token_count(opcodes) / len(new), 1.0)


def _changed_tokens_x(clean, poisoned, record):
    """Absolute size of the edit, unnormalised. Separates "small document, big edit" from "big
    document, big edit", which the change-fraction axis collapses onto the same x."""
    if (d := _diff(clean, poisoned)) is None:
        return None
    return _changed_token_count(d[2]) or None  # 0 would fall off the log axis anyway


def _first_change_x(clean, poisoned, record):
    """Where the first edit sits, as a share of the poisoned document.

    Measured on wvc this axis is nearly degenerate - LogicPoison starts editing at the top of
    almost every document (median 0.001, q3 0.005), so the answer is "at the beginning" and the
    curve collapses into one x. Kept because that is itself worth showing, but for the truncation
    question use `edit-center`, which is where the same edits actually have spread.
    """
    if (d := _diff(clean, poisoned)) is None or not d[2]:
        return None
    _old, new, opcodes = d
    return opcodes[0][3] / len(new)  # j1 of the first non-equal opcode


def _edit_center_x(clean, poisoned, record):
    """Median position of the changed tokens, as a share of the poisoned document.

    The direct test of the truncation hypothesis that the length axis can only hint at: an edit
    centred at 80% of a 4000-token document is invisible to a detector whose model reads the first
    512, however long the document is. The median rather than the first or last change, because
    LogicPoison spreads dozens of small swaps across a document - the first is always at the top
    and the last always at the bottom, while their centre of mass ranges over 0.10 to 0.92.
    """
    if (d := _diff(clean, poisoned)) is None or not d[2]:
        return None
    _old, new, opcodes = d
    positions = [j for _tag, _i1, _i2, j1, j2 in opcodes for j in range(j1, max(j2, j1 + 1))]
    return statistics.median(positions) / len(new)


def _change_hunks_x(clean, poisoned, record):
    """How many separate places the attack touched. One big rewrite and a dozen scattered entity
    swaps can carry the same changed-token count while looking nothing alike to a detector."""
    if (d := _diff(clean, poisoned)) is None:
        return None
    return len(d[2]) or None


def _length_ratio_x(clean, poisoned, record):
    """Poisoned tokens / clean tokens - does the attack inflate the document or keep its size."""
    if (d := _diff(clean, poisoned)) is None:
        return None
    old, new, _opcodes = d
    return len(new) / len(old)


def _digit_share_x(clean, poisoned, record):
    """Share of the inserted tokens that carry a digit.

    LogicPoison swaps both entities and numbers; this separates numeric poisoning from purely
    textual poisoning without needing NER. Counted on the poisoned side, so pure deletions (which
    insert nothing) contribute no tokens and a document made only of deletions drops out.
    """
    if (d := _diff(clean, poisoned)) is None:
        return None
    _old, new, opcodes = d
    inserted = [t for _tag, _i1, _i2, j1, j2 in opcodes for t in new[j1:j2]]
    if not inserted:
        return None
    return sum(any(ch.isdigit() for ch in t) for t in inserted) / len(inserted)


# Adding an axis is adding a row here: a name, a function, and what the axis is called. Everything
# derived from the diff is positives_only - a document nothing changed in has no such value, so the
# clean documents become the shared reference pool instead of populating the axis at 0.
CURVES = {
    "length": _Curve("length-performance", _length_x,
                     "Document length (characters, log scale)", "length", log_x=True),
    "change-fraction": _Curve("change-fraction", _change_fraction_x,
                              "Changed tokens / document tokens", "change-fraction",
                              positives_only=True),
    "changed-tokens": _Curve("changed-tokens", _changed_tokens_x,
                             "Changed tokens (log scale)", "edit-size",
                             log_x=True, positives_only=True),
    "first-change": _Curve("first-change", _first_change_x,
                           "Position of the first change (share of the document)", "position",
                           positives_only=True),
    "edit-center": _Curve("edit-center", _edit_center_x,
                          "Median position of the changed tokens (share of the document)",
                          "edit-position", positives_only=True),
    "change-hunks": _Curve("change-hunks", _change_hunks_x,
                           "Separate changed spans in the document (log scale)", "hunk-count",
                           log_x=True, positives_only=True),
    "length-ratio": _Curve("length-ratio", _length_ratio_x,
                           "Poisoned tokens / clean tokens", "length-ratio",
                           positives_only=True),
    "digit-share": _Curve("digit-share", _digit_share_x,
                          "Share of inserted tokens containing a digit", "digit-share",
                          positives_only=True),
}


def build_curve_series(detectors: List, curve: _Curve, clean_rows: dict, poisoned_rows: dict):
    """(series, neg_pool, n_dropped) for create_sliding_auc_chart, keyed by display name.

    Documents whose x cannot be computed are dropped and counted rather than defaulted, so a
    half-matched corpus shows up as a warning instead of as a plausible-looking curve.
    """
    series: Dict[str, List[tuple]] = {}
    neg_pool: Dict[str, List[float]] = {}
    dropped = 0
    for d in detectors:
        points, negatives = [], []
        for r in d.records:
            if curve.positives_only and not r["poisoned"]:
                negatives.append(float(r["score"]))  # shared reference pool, no x needed
                continue
            x = curve.x_of(clean_rows.get(r["doc_id"]), poisoned_rows.get(r["doc_id"]), r)
            if x is None:
                dropped += 1
                continue
            points.append((x, float(r["score"]), bool(r["poisoned"])))
        if points:
            series[display_name(d)] = points
            neg_pool[display_name(d)] = negatives
    return series, neg_pool, dropped


# --------------------------------------------------------------------------------------------
# generate_all_charts: the one call site main.py uses after run_benchmark(), redrawing every
# chart from the record files a run just wrote. Replaces the old one-chart-at-a-time CLI
# (`python -m scripts.charts <dir> --curve ... --mark ...`), which needed its own paths and mark
# values typed out by hand per dataset every time - everything here instead comes from the same
# config keys run_benchmark() itself reads, so a benchmark run and its charts never disagree
# about where things live.
# --------------------------------------------------------------------------------------------

CHART_DIR = Path("results/charts")
CURVE_DIR = CHART_DIR / "curves"  # every sliding-AUC chart, named <variable>_<dataset>_<mode>

# Where perplexity (GPT-2, 1024-token window) and NLI (DeBERTa, 512-token window) start silently
# truncating, in characters rather than tokens so the mark applies to any detector's curve on the
# same character axis (see the CURVES x_label for "length"). Measured once from GPT-2's/DeBERTa's
# tokens-per-character ratio on this project's corpora (0.267 / 0.222) - not recomputed per run,
# since that needs the actual tokenizers loaded (torch), which this module deliberately avoids.
_MODEL_TRUNCATION_MARKS = [("NLI truncates (512 tok)", 2306.0), ("perplexity truncates (1024 tok)", 3835.0)]

# The judge detectors (llm_as_a_judge/grag_as_a_judge) truncate on _JUDGE_MAX_PROMPT_CHARS
# (scripts/detector.py), a character budget rather than a tokenizer window, so unlike the marks
# above it needs no chars-per-token conversion. n-mode judges the document alone against a
# ~500-char fixed system prompt (llm_as_a_judge's; grag_as_a_judge's varies with retrieved graph
# context, so it has no single honest number, but sits close enough to be read off the same mark).
# dva-mode instead judges original-vs-new, so combined_texts() gives each side its own 40% slice
# (24000 chars) of the 60000-char budget - a different, smaller, and exact cutoff.
TRUNCATION_MARKS = {
    "regular": _MODEL_TRUNCATION_MARKS + [("LLM/GRAG judge truncates (~59.5k chars)", 59_500.0)],
    "dva": _MODEL_TRUNCATION_MARKS + [("LLM/GRAG judge truncates (24k chars, DVA)", 24_000.0)],
}


def _redraw_detect_charts(records_dir: Path, out_dir: Path, name: str, suffix: str = None) -> None:
    """detect_<name> (every metric) and detect_auc_<name> (AUC only), rebuilt from whatever record
    files are on disk. None, not [], for a mode with no records - an empty list still counts as
    "grouped" and would draw a mode legend on a chart with only one mode (graphrag_under_fire is
    insertions-only)."""
    dets_n, dets_dva = load_records(records_dir, suffix)
    if not dets_n and not dets_dva:
        return
    create_bar_chart(detectors=dets_n, detectors_edit=dets_dva or None, out_dir=out_dir, name=name)
    create_bar_chart(detectors=dets_n, detectors_edit=dets_dva or None, out_dir=out_dir,
                     name=name.replace("detect_", "detect_auc_", 1), metrics=["auc"])


def _redraw_curves(records_dir: Path, curve_stem: str, curve_dir: Path, clean_corpus: Path,
                   poisoned_corpus: Path, suffix: str = None, curve_names=None) -> None:
    """Every curve in `curve_names` (default: all of CURVES), regular and dva mode, with the
    truncation marks restored on the length curve. Silently does nothing if either corpus is
    missing, so a dataset that never got its poisoned corpus built just has no curves rather than
    crashing the rest of the redraw."""
    if not (clean_corpus.is_file() and poisoned_corpus.is_file()):
        return
    dets_n, dets_dva = load_records(records_dir, suffix)
    rows = load_corpus_rows(clean_corpus, poisoned_corpus)
    for curve_name in (curve_names or CURVES):
        curve = CURVES[curve_name]
        for mode, dets in (("regular", dets_n), ("dva", dets_dva)):
            if not dets:
                continue
            series, neg_pool, _dropped = build_curve_series(dets, curve, *rows)
            if not series:
                continue
            marks = TRUNCATION_MARKS[mode] if curve_name == "length" else None
            create_sliding_auc_chart(series, ordered_detectors(dets), curve_dir,
                                     f"{curve_name}_{curve_stem}_{mode}.png", curve,
                                     marks=marks, neg_pool=neg_pool)


def generate_all_charts(conf: dict) -> None:
    """Redraws every chart the benchmark produces: per-dataset detect charts (all metrics, plus an
    AUC-only pass), the sliding-AUC curves for every dataset with a poisoned/clean corpus pair on
    disk, the optimizer's per-detector diagnostic sweeps, and the cross-dataset parameter stability
    chart. Takes only `conf` - the same dict run_benchmark(conf) already has - and reads every path
    (records_dir, out_dir, best_params_dir, corpora) off it, the way run_benchmark itself does."""
    bconf = conf["benchmark"]
    out_dir = Path(bconf.get("out_dir", "results/charts/"))
    curve_dir = out_dir / "curves"
    records_base = Path(bconf.get("records_dir", "results/detector_records/"))
    best_params_dir = Path(bconf.get("optimize_params", {}).get("best_params_dir",
                                                                  "results/detector_best_params/"))
    poisoned_root = Path(conf.get("attack_config", {}).get("poisoned_root", "results/poisoned_data"))

    for dataset in bconf.get("datasets", []):
        _redraw_detect_charts(records_base / dataset, out_dir, f"detect_{dataset}.png")
        _redraw_curves(records_base / dataset, dataset, curve_dir,
                       Path("grag-data/input") / dataset / "corpus.jsonl",
                       poisoned_root / dataset / "corpus.jsonl")

    guf_conf = bconf.get("graphrag_under_fire", {})
    if guf_conf.get("enabled", False):
        dataset = "graphrag_under_fire"
        clean_corpus = Path("grag-data/input") / dataset / "corpus.jsonl"
        poisoned_dir = Path(guf_conf.get("poisoned_dir", f"results/poisoned_data/{dataset}"))
        for attack in guf_conf.get("attack_types", ["direct", "indirect", "enhanced"]):
            suffix = f"_{attack}"
            _redraw_detect_charts(records_base / dataset, out_dir, f"detect_{dataset}_{attack}.png",
                                  suffix=suffix)
            # Insertion-only (no paired original to diff an edit against), so only the length curve
            # is meaningful and only in regular mode - _redraw_curves already skips empty dva.
            _redraw_curves(records_base / dataset, f"{dataset}_{attack}", curve_dir, clean_corpus,
                           poisoned_dir / f"corpus_{attack}.jsonl", suffix=suffix,
                           curve_names=["length"])

    for dataset in bconf.get("logicpoison_only_datasets", []):
        records_dir = records_base / dataset
        if not (records_dir / "positive_only_summary.json").is_file():
            continue
        dets_n, dets_dva = load_records(records_dir)
        entries = ([(d, "n", d.summarize_positive_only()) for d in dets_n]
                  + [(d, "dva", d.summarize_positive_only(use_dva=True)) for d in dets_dva])
        create_positive_only_chart(entries, out_dir=out_dir, name=f"detect_{dataset}_positive_only.png")

    create_all_optimization_sweep_charts(best_params_dir, out_dir)
    create_parameter_stability_chart(best_params_dir, out_dir, records_base)
