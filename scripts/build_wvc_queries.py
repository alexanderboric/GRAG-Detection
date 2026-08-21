"""Writes queries.jsonl for the wvc corpus via LLM-generated, content-referencing questions - one
real question per article about a specific fact in its body (not just the title), so LogicPoison's
query-stage entity extraction can surface entities beyond each document's own headline subject
(the flat "What is known about {title}?" template couldn't - see conversation notes: it only ever
extracts each article's own title, making it barely different from no queries at all).

Feeds LogicPoison's real "query" stage, which needs data_root/wvc/queries.jsonl to exist.
Run: python -m scripts.build_wvc_queries
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

CORPUS_PATH = Path("grag-data/input/wvc/corpus.jsonl")
OUT_PATH = Path("datasets/logicPoison_original_Dataset/wvc/queries.jsonl")
MODEL = "mistral-small-4"
MAX_WORKERS = 4  # kept low deliberately - the wvc index build runs concurrently against the same rate-limited endpoint (concurrent_requests: 30 there); this just adds a small amount of extra load rather than risking destabilizing that multi-hour job
BODY_CHARS = 1200  # enough for a real fact, cheap enough per call

QUESTION_PROMPT = (
    "Below is a Wikipedia article. Write ONE natural question that this article answers, about a "
    "specific fact from its BODY text (a date, statistic, club, competition, relationship, or "
    "event) - NOT a generic question about the article's subject itself, and not simply restating "
    "the title. ONLY reply with the question itself, no other text.\n\n"
    "Title: {title}\n\nText: {body}"
)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
    base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
)


def generate_question(record: dict) -> dict:
    prompt = QUESTION_PROMPT.format(title=record["title"], body=record["text"][:BODY_CHARS])
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        question = resp.choices[0].message.content.strip()
    except Exception as e:
        question = f"What is known about {record['title']}?"  # fallback so one failure doesn't drop the article
        print(f"[warn] question generation failed for {record['_id']} ({record['title']}): {e}")
    return {"_id": f"q{record['_id']}", "text": question}


def main() -> None:
    with CORPUS_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, OUT_PATH.open("w", encoding="utf-8") as f:
        futures = {executor.submit(generate_question, r): r for r in records}
        for future in tqdm(as_completed(futures), total=len(records), desc="Generating wvc queries"):
            f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} queries to {OUT_PATH}")


if __name__ == "__main__":
    main()
