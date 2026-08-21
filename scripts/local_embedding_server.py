"""Local OpenAI-compatible embedding server for Qwen3-Embedding-4B, run on your own GPU instead
of the flaky university endpoint. Exposes POST /v1/embeddings matching the OpenAI schema, so
EmbeddingConnector (and GraphRAG's own internal embedding client) work against it with zero code
changes - just point EMBEDDING_BASE_URL at http://localhost:8500/v1 in .env.

Run: python local_embedding_server.py
First run downloads the model (~8GB) from HuggingFace - can take a while depending on bandwidth.
"""
import logging

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
PORT = 8500
# The existing corpus index (LanceDB vector store, settings.yaml's vector_store.vector_size) was
# built expecting 1024-dim vectors - that's what the remote endpoint was actually serving, a
# Matryoshka-truncated view of this model's native 2560-dim output. Must match, or every vector
# comparison against the existing index breaks (confirmed: native output is 2560-dim, not 1024).
TRUNCATE_DIM = 1024
# Attention memory scales O(n^2) with sequence length - confirmed the hard way: an unusually long
# WVC article (full articles aren't pre-chunked before reaching EmbeddingConnector.embed(),
# unlike GraphRAG's own 1200-token text units) triggered a single 12.17GB allocation attempt on a
# 12GB card. Capping here truncates the INPUT text itself (not just the model's positional limit)
# so a pathological document degrades to "embedded on a truncated prefix" instead of crashing the
# whole server and fragmenting CUDA memory for every request after it.
MAX_CHARS = 8000  # ~2000 tokens, generous for a single document while keeping attention bounded

print(f"Loading {MODEL_NAME} onto {'cuda' if torch.cuda.is_available() else 'cpu'}...")
# No explicit torch_dtype override here - passing model_kwargs={"torch_dtype": torch.float16}
# caused a silent native crash (no Python traceback at all) during model load, confirmed via a
# minimal reproduction without it working cleanly. Not worth chasing further tonight; the default
# dtype loads fine and 1024-dim vectors are small enough that fp16 wasn't essential for VRAM.
model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", truncate_dim=TRUNCATE_DIM)
print("Model loaded.")

app = FastAPI()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = None


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    truncated = [t[:MAX_CHARS] for t in texts]
    try:
        vectors = model.encode(truncated, convert_to_numpy=True, normalize_embeddings=True)
    except torch.OutOfMemoryError as e:
        # Release whatever PyTorch's caching allocator is holding onto so the NEXT request isn't
        # also starved by fragmentation from this one (confirmed: a single failed request left
        # only ~1GB free out of 12GB afterward, without this). Returns a clean HTTP error instead
        # of leaving the connection hanging - the client's existing retry_api_call/detect_batch
        # skip logic already knows how to handle a real API error, just not a silent OOM.
        torch.cuda.empty_cache()
        logger.error(f"OOM embedding {len(truncated)} text(s), longest={max(len(t) for t in truncated)} chars: {e}")
        raise HTTPException(status_code=503, detail=f"CUDA out of memory: {e}")
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
