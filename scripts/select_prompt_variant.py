"""Selects one fixed prompt variant per judge detector (llm_as_a_judge, grag_as_a_judge) by mean
AUC across multiple datasets - not per-dataset, since a real deployment doesn't know in advance
which attack it's facing. Writes results/detector_best_params/prompt_variants.json.

Run: python -m scripts.select_prompt_variant
"""
from pathlib import Path

from scripts.config_loader import get_config, load_config
from scripts.connectors import GraphConnector
from scripts.graphrag_manager import ensure_dataset_synced
from scripts.helpers import printv
from scripts.compare_corpus import extract_diffs, extract_unchanged, load_jsonl
from scripts.detector import grag_as_a_judge, llm_as_a_judge, prompt_variant_count, write_json


def load_val_split(dataset: str, conf: dict, val_size: int) -> tuple[list[dict], list[dict]]:
    n = conf["benchmark"]["n"]
    clean_records = load_jsonl(Path("grag-data/input") / dataset / "corpus.jsonl")
    poisoned_records = load_jsonl(Path(conf["attack_config"]["poisoned_root"]) / dataset / "corpus.jsonl")
    clean_val = extract_unchanged(clean_records, poisoned_records)[n:n + val_size]
    poisoned_val = extract_diffs(clean_records, poisoned_records)[n:n + val_size]
    return clean_val, poisoned_val


def mean_auc(cls, variant: int, datasets: list[str], conf: dict, val_size: int,
             graph_connectors: dict[str, GraphConnector] | None = None) -> float:
    aucs = []
    for dataset in datasets:
        clean_val, poisoned_val = load_val_split(dataset, conf, val_size)
        kwargs = {"graph_connector": graph_connectors[dataset]} if graph_connectors else {}
        for use_dva in (False, True):
            det = cls(variant=variant, optimize_threshold=True, **kwargs)
            det.detect_batch(clean_val, poisoned_val, use_dva=use_dva)
            aucs.append(det.roc_auc(use_dva=use_dva))
    return sum(aucs) / len(aucs)


def select_best(cls, name: str, datasets: list[str], conf: dict, val_size: int,
                 graph_connectors: dict[str, GraphConnector] | None = None) -> dict:
    best_variant, best_auc = 0, -1.0
    for variant in range(prompt_variant_count(name)):
        auc = mean_auc(cls, variant, datasets, conf, val_size, graph_connectors)
        printv(f"[{name}] variant {variant}: mean_auc={auc:.4f}", level="info")
        if auc > best_auc:
            best_variant, best_auc = variant, auc
    printv(f"[{name}] selected variant {best_variant} (mean_auc={best_auc:.4f})", level="info")
    return {"variant": best_variant, "mean_auc": best_auc}


def main() -> None:
    load_config("config.yaml")
    conf = get_config()
    sel_conf = conf["benchmark"].get("prompt_selection", {})
    datasets = sel_conf.get("datasets", ["hotpotqa", "musique"])
    val_size = sel_conf.get("val_size", 15)

    for d in datasets:
        ensure_dataset_synced(d, conf)
    graph_connectors = {d: GraphConnector(corpus=d) for d in datasets}

    results = {
        "llm_as_a_judge": select_best(llm_as_a_judge, "llm_as_a_judge", datasets, conf, val_size),
        "grag_as_a_judge": select_best(grag_as_a_judge, "grag_as_a_judge", datasets, conf, val_size, graph_connectors),
    }

    out_path = Path(conf["benchmark"].get("prompt_variants_path", "results/detector_best_params/prompt_variants.json"))
    write_json(results, out_path)
    printv(f"Wrote {out_path}", level="info")


if __name__ == "__main__":
    main()
