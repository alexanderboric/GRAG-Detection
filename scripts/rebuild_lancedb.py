"""Rebuild a graphrag output's lancedb vector stores into the schema graphrag 2.2.1 expects.

hotpotqa/musique were indexed by a newer graphrag whose lancedb tables carry
['id', 'vector', 'create_date', ...] instead of 2.2.1's ['id', 'text', 'vector', 'attributes'],
so every 2.2.1 search over them dies with KeyError: 'text'. The embeddings themselves are fine -
only the surrounding columns differ - so the text is re-joined from the parquet files by id and
the tables are rewritten. No LLM calls, no re-embedding.

  python scripts/rebuild_lancedb.py --inspect wvc          # show the reference (correct) schema
  python scripts/rebuild_lancedb.py hotpotqa musique       # rebuild in place (backs up first)
"""
import argparse
import json
import shutil
from pathlib import Path

import lancedb
import pandas as pd
import pyarrow as pa

OUTPUT_ROOT = Path(__file__).parent.parent / "graphrag-api" / "graphrag" / "output"

# lancedb table -> (parquet file, columns whose join forms the embedded text). graphrag 2.2.1
# embeds entities as "TITLE:description" and stores the embedded text again under
# attributes["title"] - both conventions are copied from the wvc store, which 2.2.1 built itself.
TABLES = {
    "default-entity-description": ("entities.parquet", ["title", "description"]),
    "default-community-full_content": ("community_reports.parquet", ["full_content"]),
    "default-text_unit-text": ("text_units.parquet", ["text"]),
}

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32())),
    pa.field("attributes", pa.string()),
])


def inspect(dataset: str) -> None:
    db = lancedb.connect(str(OUTPUT_ROOT / dataset / "lancedb"))
    for name in db.table_names():
        table = db.open_table(name)
        print(f"\n=== {name}  ({table.count_rows()} rows)")
        print("    columns:", [f.name for f in table.schema])
        row = table.to_arrow().slice(0, 1).to_pylist()
        if row:
            for key, value in row[0].items():
                shown = f"<{len(value)} floats>" if key == "vector" else repr(value)[:160]
                print(f"    {key}: {shown}")


def rebuild(dataset: str, dry_run: bool = False, force: bool = False) -> None:
    out_dir = OUTPUT_ROOT / dataset
    db_path = out_dir / "lancedb"
    backup = db_path.with_name("lancedb.pre2.2.1")
    if force and backup.exists():
        shutil.rmtree(db_path)
        shutil.copytree(backup, db_path)
        print("  restored from backup before rebuilding")
    db = lancedb.connect(str(db_path))
    existing = set(db.table_names())

    rebuilt = {}
    for name, (parquet_name, text_cols) in TABLES.items():
        if name not in existing:
            print(f"  {name}: absent, skipping")
            continue
        table = db.open_table(name)
        cols = {f.name for f in table.schema}
        if "text" in cols and "attributes" in cols:
            print(f"  {name}: already 2.2.1 schema, skipping")
            continue

        vectors = table.to_arrow().select(["id", "vector"]).to_pandas()
        source = pd.read_parquet(out_dir / parquet_name)
        for col in text_cols:
            if col not in source.columns:
                raise SystemExit(f"{parquet_name} has no '{col}' column: {list(source.columns)}")

        source["id"] = source["id"].astype(str)
        vectors["id"] = vectors["id"].astype(str)
        # hotpotqa's text_units.parquet repeats a handful of ids; without dropping them the join
        # fans out and the rebuilt store ends up with more rows than there are vectors.
        source = source.drop_duplicates(subset="id", keep="first")
        merged = vectors.merge(source[["id", *text_cols]], on="id", how="left")
        assert len(merged) == len(vectors), f"{name}: join fanned out to {len(merged)} rows"

        missing = int(merged[text_cols[0]].isna().sum())
        print(f"  {name}: {len(vectors)} vectors, {len(vectors) - missing} joined, {missing} unmatched")
        if missing == len(vectors):
            raise SystemExit(f"{name}: no ids matched {parquet_name} - id spaces differ")

        parts = [merged[c].fillna("").astype(str) for c in text_cols]
        text = parts[0] if len(parts) == 1 else parts[0].str.upper() + ":" + parts[1]
        rebuilt[name] = pa.Table.from_pydict(
            {
                "id": merged["id"].tolist(),
                "text": text.tolist(),
                "vector": [list(map(float, v)) for v in merged["vector"]],
                "attributes": [json.dumps({"title": t}) for t in text],
            },
            schema=SCHEMA,
        )

    if not rebuilt:
        print(f"[{dataset}] nothing to do")
        return
    if dry_run:
        print(f"[{dataset}] dry run - would rewrite {', '.join(rebuilt)}")
        return

    if not backup.exists():
        shutil.copytree(db_path, backup)
        print(f"  backed up to {backup.name}")
    for name, table in rebuilt.items():
        db.drop_table(name)
        db.create_table(name, table)
        print(f"  rewrote {name} ({table.num_rows} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--inspect", metavar="DATASET")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="restore the backup and rebuild again")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return
    for dataset in args.datasets:
        print(f"\n[{dataset}]")
        rebuild(dataset, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
