"""Standard error of each scenario's headline metric, from the saved detector records.

wvc and graphrag_under_fire (guf) have both classes in every record file, so the headline metric is
AUC and its standard error comes from Hanley & McNeil (1982). hotpotqa and musique run
positive-only (recall over poisoned diffs, see summarize_records.py --positive-only), so the
headline metric is recall and its standard error is the binomial SE sqrt(p(1-p)/n).

n is not fixed within a scenario: some detectors were run on a smaller pool (see hotpotqa's
llm_as_a_judge/grag_as_a_judge dva files at n=100 vs 250 elsewhere), and synthetic_dataset graphrag_
under_fire attack types differ from wvc's n=250. This reports each record file's own n rather than
a single scenario-wide number, then adds a per-scenario summary row showing SE at the min and max n
actually observed - so "standard error based on the amount of documents used" is answered exactly
rather than by a single assumed n.

    python scripts/standard_error.py
    python scripts/standard_error.py --records-dir results/detector_records
"""
import argparse
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.records import (DISPLAY_NAMES, base_name, hanley_mcneil_se, iter_record_files,
                             records_auc, tier_of)

# (scenario dir name, display label, [suffixes to iterate]). graphrag_under_fire keeps its three
# attack types in one directory distinguished by filename suffix (see records.iter_record_files).
SCENARIOS = [
    ("wvc", "WVC", [None]),
    ("hotpotqa", "HotpotQA", [None]),
    ("musique", "MuSiQue", [None]),
    ("graphrag_under_fire", "GUF", ["direct", "indirect", "enhanced"]),
]


def binomial_se(p: float, n: int) -> float:
    if n < 1:
        return float("nan")
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def rows_for(records_dir: Path, subdir: str, suffixes: list) -> list:
    rows = []
    for suffix in suffixes:
        path = records_dir / subdir
        if not path.exists():
            continue
        for name, mode, records, _threshold, _coverage in iter_record_files(path, suffix):
            n_pos = sum(1 for r in records if r["poisoned"])
            n_neg = len(records) - n_pos
            base = DISPLAY_NAMES.get(base_name(name), base_name(name))
            _rank, tier = tier_of(name)
            if n_neg == 0:
                p = sum(1 for r in records if r["score"] > _threshold) / n_pos if n_pos else float("nan")
                rows.append({"detector": base, "tier": tier, "mode": mode, "group": suffix,
                            "n": len(records), "metric": "recall", "value": p,
                            "se": binomial_se(p, n_pos)})
            else:
                value = records_auc(records)
                se = hanley_mcneil_se(value, n_pos, n_neg) if value is not None else float("nan")
                rows.append({"detector": base, "tier": tier, "mode": mode, "group": suffix,
                            "n": len(records), "metric": "auc", "value": value, "se": se})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records-dir", type=Path, default=Path("results/detector_records"))
    args = parser.parse_args()

    for subdir, label, suffixes in SCENARIOS:
        rows = rows_for(args.records_dir, subdir, suffixes)
        if not rows:
            print(f"\n{label}: no record files in {args.records_dir / subdir}")
            continue

        rows.sort(key=lambda r: (r["tier"][0], r["group"] or "", r["detector"], r["mode"]))
        header = f"{'detector':26s} {'mode':5s} {'group':10s} {'n':>5s} {'metric':>7s} {'value':>7s} {'SE':>7s}"
        print(f"\n{label}  ({args.records_dir / subdir})")
        print(header)
        print("-" * len(header))
        tier = None
        for r in rows:
            if r["tier"] != tier:
                tier = r["tier"]
                print(f"[{tier[1]}]")
            val = f"{r['value']:.3f}" if r["value"] == r["value"] else "n/a"
            se = f"{r['se']:.3f}" if r["se"] == r["se"] else "n/a"
            print(f"{r['detector']:26s} {r['mode']:5s} {(r['group'] or '-'):10s} {r['n']:5d} "
                  f"{r['metric']:>7s} {val:>7s} {se:>7s}")

        ns = [r["n"] for r in rows]
        n_lo, n_hi = min(ns), max(ns)
        auc_rows = [r for r in rows if r["metric"] == "auc"]
        if auc_rows:
            se_lo = hanley_mcneil_se(0.5, n_hi // 2, n_hi - n_hi // 2)
            se_hi = hanley_mcneil_se(0.5, n_lo // 2, n_lo - n_lo // 2)
            print(f"\nAUC SE at chance (0.5): ~{se_lo:.3f} at n={n_hi} (balanced), "
                 f"~{se_hi:.3f} at n={n_lo} (balanced) - narrower n gives noisier AUC.")
        else:
            se_lo = binomial_se(0.5, n_hi)
            se_hi = binomial_se(0.5, n_lo)
            print(f"\nRecall SE at p=0.5: ~{se_lo:.3f} at n={n_hi}, ~{se_hi:.3f} at n={n_lo} "
                 "- narrower n gives noisier recall.")


if __name__ == "__main__":
    main()
