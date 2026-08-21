# Entry point: loads config, then dispatches to graphrag_test / visualize_graph / attack / benchmark.

from dotenv import load_dotenv
load_dotenv()  # load .env (incl. OpenAI-compatible config for LogicPoison)

import asyncio
import os

from scripts.helpers import printv
from scripts.config_loader import load_config, get_config
from scripts.graphrag_manager import check_index_status, run_indexing, query, load_graph, ensure_dataset_synced, set_active_corpus
from scripts.attack import run_attack
from scripts.benchmark import run_benchmark
from scripts.charts import generate_all_charts

_ATTACK_PATH_KEYS = ["data_root", "corpus_entities_root", "queries_entities_root", "poisoned_root"]


def visualize_graph(G):
    import webbrowser
    from pyvis.network import Network
    net = Network(notebook=False)
    net.from_nx(G)
    net.write_html("graph.html")
    webbrowser.open("file://" + os.path.abspath("graph.html"))


async def main():
    load_config("config.yaml")
    conf = get_config()
    printv("Config loaded successfully.")

    project_root = os.path.dirname(os.path.abspath(__file__))
    if "attack_config" in conf:
        for path_key in _ATTACK_PATH_KEYS:
            path = conf["attack_config"].get(path_key)
            if path and not os.path.isabs(path):
                conf["attack_config"][path_key] = os.path.join(project_root, path)

    if conf["graphrag_test"]:
        await graphrag_test()
    if conf["visualize_graph"]:
        visualize_graph(load_graph())
    if conf["attack"]:
        run_attack(conf)
    if conf.get("benchmark", {}).get("run", False):
        await run_benchmark(conf)
        generate_all_charts(conf)


async def graphrag_test():
    """Smoke-tests the GraphRAG pipeline: indexes the first configured dataset if needed, then
    runs one local query against it."""
    conf = get_config()
    datasets = conf.get("benchmark", {}).get("datasets", [])
    if datasets:
        ensure_dataset_synced(datasets[0], conf)
        set_active_corpus(datasets[0])

    status = check_index_status()
    if status["is_complete"]:
        printv("GraphRAG index is complete.", level="info")
    else:
        printv(f"Index incomplete (missing: {status['missing_files']}). Starting indexing pipeline...", level="warning")
        if not await run_indexing():
            printv("Indexing failed. Check app.log for details.", level="error")
            return
        printv("Indexing completed successfully.", level="info")

    response = await query("What are the main topics in the documents?", method="local")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
