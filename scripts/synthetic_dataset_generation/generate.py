"""Generation script that produced the synthetic_dataset dataset (350 docs / 200 diffs) actually
shipped in grag-data/input/synthetic_dataset/{corpus.jsonl,diffs.jsonl}. Recovered from the agent
session transcript and committed here for reproducibility - was originally run from scratchpad,
so paths/behavior are unchanged from that run except this docstring and the world_bible import
below. Scales the original 100-doc/50-diff run up ~3.5x using world_bible's 70 entities.

Generates (into this same directory - copy corpus_v2.jsonl/diffs_v2.jsonl to
grag-data/input/synthetic_dataset/{corpus.jsonl,diffs.jsonl} to actually use them):
  - corpus_v2.jsonl: 350 short (~100-300 word) documents ({_id, title, text} shape).
  - diffs_v2.jsonl: 200 edit pairs (100 legit, 100 poisoned), WVC diff-dict shape.

Same generation discipline as v1: identical prompt template for both edit classes, only the
correctness instruction differs, so the only systematic difference between classes is factual
correctness, never a style/length artifact (see the project's own grag_contradiction length-
confound note in config.yaml). Uses the project's existing OpenAI-compatible client pattern
(scripts/build_prompts.py): OPENAI_API_KEY / OPENAI_BASE_URL from .env, model "mistral-small-4".

Resumable throughout (docs cache + edits cache, both append-only JSONL) since ~270 sequential
calls (70 doc-batch calls + 200 edit calls) is long enough that a transient failure partway
through would otherwise cost everything before it.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from world_bible import ENTITIES, WORLD_SUMMARY

# load_dotenv() with no path walks up from THIS FILE's own directory, never reaching the repo's
# .env when run from outside the repo - point at it explicitly, works from here too.
load_dotenv(Path("/Users/alexanderboric/Desktop/Uni/Bachelor/GraphPoison/.env"))

MODEL = "mistral-small-4"
OUT_DIR = Path(__file__).parent
SEED = 20260819

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
    base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
    timeout=180,
    max_retries=0,  # call_json() does its own retry loop with logging
)

rng = random.Random(SEED)


def call_json(prompt: str, retries: int = 8) -> dict:
    """More retries + real exponential backoff than v1's flat 3s (4 tries) - v1 only hit 429s
    during the concurrent benchmark run, not sequential generation, but at ~4x the call volume
    here a shared-endpoint throttling window during generation itself is more plausible, so this
    is sized to actually ride one out (up to ~2 minutes of backoff) rather than give up fast."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  retry {attempt+1}/{retries} after error: {e!r} - waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {last_err}")


DOC_PROMPT = """You are writing short encyclopedia-style documents about an ENTIRELY FICTIONAL, \
invented world. None of these people, places, organizations, or events are real - they must not \
correspond to anything in the real world. This is important: the whole point is that the content \
cannot exist in any language model's training data.

Full world reference (use this to stay internally consistent and to naturally cross-reference \
other entities by their exact names where relevant):
---
{world_summary}
---

Now write {n} SEPARATE short documents about "{entity_name}" ({entity_type}), one for each of these \
facets: {facets}.

Ground facet doc requirements:
- Each document is 100-300 words, self-contained, reads like a plain encyclopedia or news article \
paragraph (no headers, no bullet points, no facet name mentioned literally).
- Each document must be consistent with these canonical facts about {entity_name}: {facts}
- Naturally weave in cross-references to other entities from the world reference above where \
relevant (using their exact names), so the corpus reads as one coherent world, but do not force \
irrelevant references into every document.
- Plain ASCII punctuation only.
- Do not repeat the same sentences verbatim across the {n} documents - vary the angle per facet.

Reply with ONLY a JSON object of this exact shape:
{{"docs": [{{"facet": "<facet_name>", "title": "<short title>", "text": "<100-300 word document>"}}, ...]}}"""


DOCS_CACHE = OUT_DIR / "_docs_cache_v2.jsonl"


def generate_docs() -> list[dict]:
    done: dict[str, list[dict]] = {}
    if DOCS_CACHE.exists():
        with DOCS_CACHE.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["entity_key"]] = row["docs"]
        print(f"[cache] resuming - {len(done)}/{len(ENTITIES)} entities already done")

    for i, (key, ent) in enumerate(ENTITIES.items()):
        if key in done:
            continue
        print(f"[docs {i+1}/{len(ENTITIES)}] {ent['name']} ({len(ent['facets'])} facets)...")
        prompt = DOC_PROMPT.format(
            world_summary=WORLD_SUMMARY,
            n=len(ent["facets"]),
            entity_name=ent["name"],
            entity_type=ent["type"],
            facets=", ".join(ent["facets"]),
            facts=" ".join(ent["facts"]),
        )
        data = call_json(prompt)
        docs = data["docs"]
        if len(docs) != len(ent["facets"]):
            print(f"  WARNING: expected {len(ent['facets'])} docs, got {len(docs)}")
        done[key] = docs
        with DOCS_CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"entity_key": key, "docs": docs}, ensure_ascii=False) + "\n")

    records = []
    idx = 0
    for key, ent in ENTITIES.items():
        for d in done[key]:
            records.append({
                "_id": str(idx),
                "title": d.get("title", f"{ent['name']} ({d.get('facet','')})"),
                "text": d["text"].replace("�", "-"),
                "_entity_key": key,
                "_facet": d.get("facet", ""),
            })
            idx += 1
    return records


EDIT_PROMPT = """You are producing a test case for a fact-checking research benchmark, over an \
ENTIRELY FICTIONAL invented world (see reference below - nothing here is real).

World reference:
---
{world_summary}
---

Here is one existing document from this fictional corpus, about "{entity_name}":
---
{original_text}
---

Task: produce ONE edited version of this document. {instruction}

Strict requirements so the edited version is comparable across test cases regardless of which \
kind of edit this is:
- Keep the edit to a SINGLE localized change (one fact/clause), everything else word-for-word identical.
- The edited document's total length must stay within about 5% of the original's word count.
- Same tone, register, and sentence structure as the original - a reader should not be able to tell \
from STYLE ALONE what kind of edit this is; only the factual content differs.
- Plain ASCII punctuation only.
- Return the FULL edited document text (not just the changed sentence).

Reply with ONLY a JSON object: {{"new_text": "<full edited document text>"}}"""

LEGIT_INSTRUCTION = (
    "Make a genuine, benign, LEGITIMATE update or correction - e.g. a small clarification, an "
    "update consistent with the rest of the fictional world's established facts (the world "
    "reference above), or a harmless stylistic-but-factual refinement. The result must remain "
    "fully consistent with the world reference - it must NOT contradict any fact stated there."
)
POISON_INSTRUCTION = (
    "Introduce ONE subtly WRONG fact that contradicts the world reference above - e.g. swap a "
    "name, number, date, or relationship for an incorrect one (you may reuse another entity's "
    "name/number/date from the world reference to make it plausible-sounding, or invent a "
    "similarly-styled false value). The change must read fluently and be subtle (not an obvious "
    "absurdity), but it must genuinely contradict the world reference's established facts."
)


EDITS_CACHE = OUT_DIR / "_edits_cache_v2.jsonl"


def generate_edits(corpus: list[dict], n_legit: int, n_poison: int) -> list[dict]:
    pool = corpus[:]
    rng.shuffle(pool)
    legit_src = pool[:n_legit]
    poison_src = pool[n_legit:n_legit + n_poison]

    done: dict[tuple[str, str], dict] = {}
    if EDITS_CACHE.exists():
        with EDITS_CACHE.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[(row["record_id"], "poison" if row["is_vandalism"] else "legit")] = row
        print(f"[cache] resuming - {len(done)}/{n_legit+n_poison} edits already done")

    editid = 0
    total = n_legit + n_poison
    n_done_at_start = len(done)
    for is_poison, src_list in ((False, legit_src), (True, poison_src)):
        label = "poison" if is_poison else "legit"
        for rec in src_list:
            key = (rec["_id"], label)
            if key in done:
                continue
            entity_name = ENTITIES[rec["_entity_key"]]["name"]
            n_done_at_start += 1
            print(f"[edits {n_done_at_start}/{total} {label}] doc _id={rec['_id']} ({entity_name})...")
            prompt = EDIT_PROMPT.format(
                world_summary=WORLD_SUMMARY,
                entity_name=entity_name,
                original_text=rec["text"],
                instruction=POISON_INSTRUCTION if is_poison else LEGIT_INSTRUCTION,
            )
            data = call_json(prompt)
            new_text = data["new_text"].replace("�", "-")
            row = {
                "original_text": rec["text"],
                "new_text": new_text,
                "editid": f"mp_{editid}",
                "article_title": rec["title"],
                "old_revision_id": f"mp_{rec['_id']}_orig",
                "new_revision_id": f"mp_{rec['_id']}_{label}",
                "is_vandalism": is_poison,
                "record_id": rec["_id"],  # kept as str here; cast to int on final write
            }
            done[key] = row
            with EDITS_CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            editid += 1

    diffs = []
    for is_poison, src_list in ((False, legit_src), (True, poison_src)):
        label = "poison" if is_poison else "legit"
        for rec in src_list:
            row = dict(done[(rec["_id"], label)])
            row["record_id"] = int(row["record_id"])
            diffs.append(row)
    return diffs


def main():
    n_entities = len(ENTITIES)
    n_docs_target = sum(len(e["facets"]) for e in ENTITIES.values())
    print(f"=== Generating {n_docs_target} fictional documents ({n_entities} entities) ===")
    corpus = generate_docs()
    assert len(corpus) == n_docs_target, f"expected {n_docs_target} docs, got {len(corpus)}"
    corpus_path = OUT_DIR / "corpus_v2.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus)} docs -> {corpus_path}")

    n_legit, n_poison = 100, 100
    print(f"\n=== Generating {n_legit} legit + {n_poison} poisoned edit pairs ===")
    diffs = generate_edits(corpus, n_legit=n_legit, n_poison=n_poison)
    diffs_path = OUT_DIR / "diffs_v2.jsonl"
    with diffs_path.open("w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(diffs)} diffs -> {diffs_path}")


if __name__ == "__main__":
    main()
