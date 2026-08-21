"""Prints a comparison table straight from saved detector record files.

Reads the JSON that detector.save_to_file() writes, so it needs no GPU, no servers and no detector
objects - just the scores and labels (scripts/records.py does the reading and the arithmetic, and is
shared with scripts/charts.py so table and charts can never disagree). Groups by awareness tier
(document-level / embedding-aware / grag-aware) in the same order as the charts, shows n and dva
side by side, and reports each grag-aware detector's fallback rate next to its AUC, since an AUC
earned mostly through the no-evidence fallback is not a graph result.

  python scripts/summarize_records.py results/detector_records/wvc
  python scripts/summarize_records.py results/detector_records/musique --positive-only
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file path rather than `python -m scripts.summarize_records`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.records import DISPLAY_NAMES, base_name, iter_record_files, metrics_from_records, tier_of


def fallback_note(coverage: dict | None) -> str:
    if not coverage or not coverage.get("n_total"):
        return ""
    n_fb = coverage.get("n_fallback", 0)
    if not n_fb:
        return "graph 100%"
    pct = 100 * n_fb / coverage["n_total"]
    pois = coverage.get("n_fallback_poisoned", 0)
    clean = coverage.get("n_fallback_clean", 0)
    return f"fallback {pct:.0f}% ({pois}P/{clean}C)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--positive-only", action="store_true",
                        help="no clean class present - report recall only, as the logicpoison leg does")
    args = parser.parse_args()

    rows = []
    for name, mode, records, threshold, coverage in iter_record_files(args.records_dir):
        rank, tier = tier_of(name)
        base = base_name(name)
        rows.append({
            "rank": rank, "tier": tier, "name": DISPLAY_NAMES.get(base, base),
            "full": name, "mode": mode, "note": fallback_note(coverage),
            **metrics_from_records(records, threshold),
        })

    if not rows:
        print(f"no record files in {args.records_dir}")
        return

    rows.sort(key=lambda r: (r["rank"], r["name"], r["mode"]))
    tier = None
    if args.positive_only:
        header = f"{'detector':26s} {'mode':5s} {'n':>5s} {'recall':>8s}  note"
    else:
        header = f"{'detector':26s} {'mode':5s} {'n':>5s} {'AUC':>7s} {'informed':>9s} {'MCC':>7s}  note"
    print(f"\n{args.records_dir}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["tier"] != tier:
            tier = r["tier"]
            print(f"\n[{tier}]")
        if args.positive_only:
            print(f"{r['name']:26s} {r['mode']:5s} {r['n']:5d} {r['recall']:8.3f}  {r['note']}")
        else:
            auc = f"{r['auc']:.3f}" if r["auc"] is not None else "n/a"
            print(f"{r['name']:26s} {r['mode']:5s} {r['n']:5d} {auc:>7s} "
                  f"{r['informedness']:9.3f} {r['mcc']:7.3f}  {r['note']}")
    print("\nAUC noise floor at n=60+60 is sd~0.051, at n=100+100 ~0.040 - "
          "differences under ~0.10 are not meaningful.")


if __name__ == "__main__":
    main()
