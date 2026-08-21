import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
corpus = [json.loads(l) for l in (OUT_DIR / "corpus.jsonl").open(encoding="utf-8")]
diffs = [json.loads(l) for l in (OUT_DIR / "diffs.jsonl").open(encoding="utf-8")]

print(f"corpus: {len(corpus)} docs")
lens = [len(d["text"].split()) for d in corpus]
print(f"  word counts: min={min(lens)} max={max(lens)} avg={sum(lens)/len(lens):.0f}")
ids = [d["_id"] for d in corpus]
assert ids == [str(i) for i in range(len(corpus))], "corpus _id not sequential 0..N-1"
print("  _id sequential: OK")

print(f"\ndiffs: {len(diffs)} total, legit={sum(1 for d in diffs if not d['is_vandalism'])}, poison={sum(1 for d in diffs if d['is_vandalism'])}")
for d in diffs:
    ow, nw = len(d["original_text"].split()), len(d["new_text"].split())
    ratio = nw / ow if ow else 0
    if not (0.85 <= ratio <= 1.15):
        print(f"  WARN length drift record_id={d['record_id']} vandalism={d['is_vandalism']} orig={ow}w new={nw}w ratio={ratio:.2f}")
    if d["original_text"] == d["new_text"]:
        print(f"  WARN no-op edit record_id={d['record_id']}")

record_ids = [d["record_id"] for d in diffs]
assert len(record_ids) == len(set(record_ids)), "duplicate record_id across diffs"
print("  record_id uniqueness: OK")
for d in diffs:
    assert 0 <= d["record_id"] < len(corpus)
print("  record_id range: OK")

print("\n=== 3 legit examples ===")
for d in [x for x in diffs if not x["is_vandalism"]][:3]:
    print(f"--- record_id={d['record_id']} ({d['article_title']}) ---")
    print("ORIG:", d["original_text"][:400])
    print("NEW: ", d["new_text"][:400])
    print()

print("=== 3 poison examples ===")
for d in [x for x in diffs if x["is_vandalism"]][:3]:
    print(f"--- record_id={d['record_id']} ({d['article_title']}) ---")
    print("ORIG:", d["original_text"][:400])
    print("NEW: ", d["new_text"][:400])
    print()
