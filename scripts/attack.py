"""Dispatches the LogicPoison attack pipeline as a subprocess, driven by config.yaml's attack_config."""

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.helpers import printv

# CLI flags forwarded to LogicPoison's main.py as-is, keyed by attack_config field name.
_PATH_ARGS = ["data_root", "corpus_entities_root", "queries_entities_root", "poisoned_root"]
_SCALAR_ARGS = ["batch_size", "max_workers", "queue_factor"]


def ensure_queries_entities(attack_config: dict) -> None:
    """Synthesizes a queries_entities JSONL from corpus entity stats for datasets that skipped the
    query stage, since LogicPoison's logic stage requires that file to exist.

    Requires corpus_entities/<dataset>.json to already be on disk - see run_attack(), which runs
    the "global" stage as its own pre-pass before calling this exactly so that's true even for a
    dataset attacked for the first time. Without that ordering this used to write an empty stub
    whenever corpus_entities didn't exist yet, which silently poisoned nothing at all:
    logic_poison.py's build_replace_map() only builds a replacement pool for entity types present
    in queries_entities, so an empty file meant zero types (and therefore zero replacements) even
    though the corpus's own top-frequency entity pools were populated and ready to poison.
    """
    qe_root = Path(attack_config.get("queries_entities_root", "results/queries_entities"))
    ce_root = Path(attack_config.get("corpus_entities_root", "results/corpus_entities"))

    for dataset in attack_config.get("datasets", []):
        qe_path = qe_root / f"{dataset}.jsonl"
        ce_path = ce_root / f"{dataset}.json"
        if qe_path.exists():
            continue
        qe_path.parent.mkdir(parents=True, exist_ok=True)
        if not ce_path.exists():
            # Last resort so the "logic" stage doesn't crash on a missing file - but this means
            # zero poisoning for this dataset (see docstring). Shouldn't happen in practice since
            # run_attack() runs "global" before calling here whenever it's in the requested stages.
            qe_path.write_text("", encoding="utf-8")
            printv(f"No corpus entities for {dataset} yet - wrote empty queries_entities "
                   f"(this dataset will not be poisoned). Run the 'global' stage first.", level="warning")
            continue

        corpus_ents = json.loads(ce_path.read_text(encoding="utf-8"))
        entities_list = [
            {"hop": 1, "entity": entity, "type": etype}
            for etype, pairs in corpus_ents.items()
            for entity, _ in pairs[:3]  # top-3 per type
        ]
        record = {"_id": "q_synthetic_0", "text": "synthetic query from corpus", "entities": entities_list, "error": None}
        with qe_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        printv(f"Synthesised queries_entities for {dataset} ({len(entities_list)} entities from corpus stats).", level="info")


def ensure_datasets_visible(attack_config: dict, project_root: str) -> None:
    """Symlinks any requested dataset into data_root if LogicPoison can't already see it there.

    LogicPoison's own main.py lists datasets by scanning data_root's subdirectories, so any
    dataset kept outside the logicPoison_original_Dataset submodule (wvc, graphrag_under_fire,
    synthetic_dataset, ...) is invisible to it as-is. ensure_dataset_synced() in
    graphrag_manager.py already has a fallback for the GraphRAG-input side of this
    (datasets/<dataset>/ when data_root doesn't have it) - this is the same idea for the attack
    side, checked one level further: datasets/<dataset>/ first, then grag-data/input/<dataset>/
    (where a dataset's corpus.jsonl may already live with nothing under datasets/ at all, e.g.
    synthetic_dataset's LLM-generated corpus).
    """
    data_root = attack_config.get("data_root")
    if not data_root:
        return
    for dataset in attack_config.get("datasets", []):
        if dataset == "all":
            continue
        link_path = os.path.join(data_root, dataset)
        if os.path.exists(link_path):
            continue
        for candidate in (os.path.join(project_root, "datasets", dataset),
                          os.path.join(project_root, "grag-data", "input", dataset)):
            if os.path.isfile(os.path.join(candidate, "corpus.jsonl")):
                os.makedirs(data_root, exist_ok=True)
                os.symlink(candidate, link_path)
                printv(f"Linked {dataset} into {data_root} (source: {candidate}).", level="info")
                break
        else:
            printv(f"No corpus.jsonl found for dataset '{dataset}' under datasets/ or "
                   f"grag-data/input/ - LogicPoison will report it unknown.", level="warning")


def _build_args(attack_config: dict, stages: list) -> list:
    args = [sys.executable, "-u", "main.py", "--stages"] + stages
    if "datasets" in attack_config:
        args.extend(["--datasets"] + attack_config["datasets"])
    for key in _PATH_ARGS + _SCALAR_ARGS:
        if key in attack_config:
            args.extend([f"--{key}", str(attack_config[key])])
    if "query_model" in attack_config:
        args.extend(["--query_model", str(attack_config["query_model"])])
    if "top_ratio" in attack_config:
        args.extend(["--top_ratio", str(attack_config["top_ratio"])])
    if attack_config.get("force", False):
        args.append("--force")
    return args


def run_attack(conf: dict) -> None:
    """Runs LogicPoison as a subprocess in its own directory. Assumes attack_config's paths are
    already absolute (main() converts them once at startup before dispatching here)."""
    attack_config = conf["attack_config"].copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ensure_datasets_visible(attack_config, project_root)

    logicpoison_dir = os.path.join(project_root, "attacks", "logicPoison")
    stages = attack_config.get("stages", ["all"])

    # "global" is run as its own pass first, synchronously, whenever requested - not because the
    # combined call below couldn't run it too (it will, LogicPoison's own stage ordering already
    # runs global before logic), but because ensure_queries_entities() needs corpus_entities/
    # <dataset>.json to already exist for a dataset attacked for the first time, and that file is
    # exactly what "global" produces. Running the full requested stage list in one call meant
    # ensure_queries_entities() always ran before global had ever produced anything, silently
    # poisoning nothing (see its docstring).
    if "global" in stages or "all" in stages:
        try:
            subprocess.run(_build_args(attack_config, ["global"]), cwd=logicpoison_dir,
                           check=True, env=os.environ.copy())
        except subprocess.CalledProcessError as e:
            printv(f"Attack pipeline's global stage failed with exit code {e.returncode}", level="error")
            return

    ensure_queries_entities(attack_config)

    try:
        subprocess.run(_build_args(attack_config, stages), cwd=logicpoison_dir,
                       check=True, env=os.environ.copy())
    except subprocess.CalledProcessError as e:
        printv(f"Attack pipeline failed with exit code {e.returncode}", level="error")
