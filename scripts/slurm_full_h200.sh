#!/bin/bash
#SBATCH --job-name=graph_benchmark_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

# Full-card H200 run of config.yaml as-is, in place (this directory, not an isolated clone):
# wvc's own optimize+benchmark loop (benchmark.datasets), graphrag_under_fire (its own enabled
# flag), and musique/hotpotqa (logicpoison_only_datasets) all run unconditionally one after
# another from a single `python main.py` - see run_benchmark()/benchmark.py:469-470. Unlike
# slurm_h200.sh/slurm_guf.sh (which each run one slice of this from a separate directory with a
# trimmed config.yaml so two jobs sharing API_PORT=8000 don't kill each other), this is meant to
# be the only job running against this directory.
#
# Full card (143GB), not the gpuh200mini 2g.35gb/1g.35gb MIG pair: the MIG slice's KV cache only
# fits ~2 concurrent map calls at 16384 tokens (graphrag's global search fires up to
# concurrent_requests=25), serializing a single document's map phase into ~12 rounds. A full H200
# measured ~400k KV-cache tokens (~24x concurrency) on graphrag_under_fire's indexing run instead.
#
# musique/hotpotqa currently have parquets (entities/communities/community_reports - the
# expensive LLM-driven stages) but no lancedb/ at all, so graphrag_init() will run a full
# from-scratch reindex for each the first time this touches them (check_index_status() sees them
# as incomplete) - budget the 2-day walltime accordingly.

set -euo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

module --force purge
module load palma/2024a GCCcore/13.3.0 Python/3.12.3
module load CUDA/13.0.2

if [ ! -e .venv ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt 'graphrag==2.2.1' python-dotenv PyYAML -q
    deactivate
fi

if [ ! -e .venv-vllm ]; then
    python3 -m venv .venv-vllm
    source .venv-vllm/bin/activate
    pip install --upgrade pip -q
    pip install vllm ninja -q
    deactivate
fi

source .venv/bin/activate

# Point graphrag's own local/global/basic-search LLM+embedding calls at our local servers - see
# slurm_wvc_optimize.sh for why this can't just be OPENAI_BASE_URL/EMBEDDING_BASE_URL env vars.
python3 - <<'PY'
import yaml
from pathlib import Path

path = Path("graphrag-api/graphrag/settings.yaml")
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
for model_key in ("default_chat_model", "default_embedding_model"):
    model_cfg = cfg["models"][model_key]
    model_cfg["api_base"] = "http://localhost:8001/v1" if model_key == "default_chat_model" else "http://localhost:8500/v1"
    model_cfg["parallelization"] = {"stagger": 0.0, "num_threads": 16}
    # graphrag's 50000 TPM / 1000 RPM cloud-quota defaults are meaningless against our own vLLM
    # server; fnllm skips building the RPM/TPM limiters when these are falsy.
    model_cfg["tokens_per_minute"] = 0
    model_cfg["requests_per_minute"] = 0
    if model_key == "default_chat_model":
        # A full H200 measured ~24x concurrency at 16384 tokens/request (400k KV-cache tokens);
        # 20 leaves headroom rather than sitting right at graphrag's default concurrent_requests=25.
        model_cfg["concurrent_requests"] = 20
gs = cfg.setdefault("global_search", {})
gs["map_max_length"] = 300
gs["dynamic_search_use_summary"] = True
path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("Patched graphrag-api settings.yaml: api_base -> local servers, parallelization loosened, TPM/RPM limits removed, global-search map_max_length=300 + summary-based dynamic selection")
PY

# Single full GPU - both servers share device 0, no MIG slice detection needed.
LLM_GPU=0
EMB_GPU=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "LLM judge and embedding server both on GPU $LLM_GPU (full H200)"

CUDA_VISIBLE_DEVICES=$EMB_GPU python scripts/local_embedding_server.py > logs/embedding_server_${SLURM_JOB_ID}.log 2>&1 &
EMB_PID=$!

# --max-model-len 32768: graphrag counts its global-search map context in cl100k
# (settings.yaml encoding_model) while the server enforces Qwen2.5 tokens - the two disagree by
# ~1.3-1.4x, so a nominally-13000-token context 400s on arrival at 16384. See slurm_h200.sh.
PATH="$SLURM_SUBMIT_DIR/.venv-vllm/bin:$PATH" CUDA_VISIBLE_DEVICES=$LLM_GPU .venv-vllm/bin/vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq_marlin \
    --served-model-name mistral-small-4 \
    --port 8001 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 32768 \
    > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
LLM_PID=$!

cleanup() {
    kill "$EMB_PID" "$LLM_PID" 2>/dev/null || true
    wait "$EMB_PID" "$LLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for() {
    local url=$1 name=$2 timeout=$3
    local waited=0
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [[ $waited -ge $timeout ]]; then
            echo "ERROR: $name did not become ready within ${timeout}s" >&2
            exit 1
        fi
    done
    echo "$name ready after ${waited}s"
}

wait_for "http://localhost:8500/v1/models" "embedding server" 600
wait_for "http://localhost:8001/v1/models" "vLLM judge server" 1200

export OPENAI_BASE_URL="http://localhost:8001/v1"
export OPENAI_API_KEY="local"
export EMBEDDING_BASE_URL="http://localhost:8500/v1"
export EMBEDDING_API_KEY="local"
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export GRAPHRAG_API_KEY="local"

# Only one GPU here, so main.py's own local models (NLIScorer/perplexity) share it with vLLM -
# 0.70 (not 0.92) leaves them headroom. This job also reindexes musique/hotpotqa from scratch, and
# graphrag's generate_text_embeddings workflow drives the embedding server hard enough to OOM at
# higher vLLM utilization (see slurm_guf.sh, job 46067565) - 0.70 leaves ~43GB for the embedding
# server and main.py's local models, still giving vLLM ~79GB of KV cache (~9x concurrency at 32768).
export CUDA_VISIBLE_DEVICES=$LLM_GPU

python -u main.py
