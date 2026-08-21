#!/usr/bin/env python3
"""Convert GraphRAG input documents to logicPoison dataset format."""
# THis is only used for own files, the datasets in /datasets are already in the correct format by default

import json
import os
import random
import re
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> list:
    """Split text into chunks with overlap by sentences.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Overlap between chunks to maintain context
    """
    # Split by sentence (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out very short sentences and metadata
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        potential = current_chunk + " " + sentence if current_chunk else sentence
        
        if len(potential) > chunk_size and current_chunk:
            # Save current chunk and start new one with overlap
            chunks.append(current_chunk.strip())
            # Keep last overlap_size characters for context
            current_chunk = current_chunk[-overlap:] + " " + sentence if overlap > 0 else sentence
        else:
            current_chunk = potential
    
    if current_chunk and len(current_chunk.strip()) > 50:
        chunks.append(current_chunk.strip())
    
    return [c for c in chunks if len(c.strip()) > 50]


def create_corpus_jsonl(text_files: list, output_path: str):
    """Create corpus.jsonl from text files."""
    with open(output_path, "w", encoding="utf-8") as f:
        total_chunks = 0
        for text_file in text_files:
            print(f"Processing: {text_file}")
            with open(text_file, "r", encoding="utf-8") as tf:
                content = tf.read()
            
            # Clean up text
            content = content.strip()
            if not content:
                continue
            
            # Remove common metadata patterns (ISBN, URLs, copyright info)
            content = re.sub(r'ISBN.*?\n', '', content)
            content = re.sub(r'www\.\S+', '', content)
            content = re.sub(r'©.*?\n', '', content)
            content = re.sub(r'http\S+', '', content)
            
            # Chunk the text with better parameters
            chunks = chunk_text(content, chunk_size=1500, overlap=300)
            
            for chunk in chunks:
                if chunk.strip() and len(chunk.strip()) > 50:
                    record = {"text": chunk}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1
    
    print(f"[OK] Created corpus.jsonl with {total_chunks} documents")


def create_queries_jsonl(output_path: str, num_queries: int = 5):
    """Create sample queries.jsonl."""
    sample_queries = [
        "What are the main topics discussed in the documents?",
        "Who are the important people mentioned?",
        "What is the historical context?",
        "What are the key concepts explained?",
        "What is the narrative or story about?",
        "What philosophical ideas are presented?",
        "What events or situations are described?",
        "What relationships or connections are mentioned?",
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, query in enumerate(sample_queries[:num_queries]):
            record = {"_id": f"q{i+1}", "text": query}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[OK] Created queries.jsonl with {num_queries} sample queries")


def _write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


ATTACK_TYPES = ("direct", "indirect", "enhanced")


def ingest_graphrag_under_fire_corpus(
    para_path: str | Path = "datasets/graphragUnderFire/need_save_para.txt",
    clean_output: str | Path = "datasets/graphrag_under_fire/corpus.jsonl",
) -> None:
    """Reads the "GraphRAG Under Fire" paper's clean corpus (one paragraph per
    non-empty line in need_save_para.txt) and writes it out as corpus.jsonl in
    the {_id, title, text} shape the rest of the pipeline expects (same as
    datasets/logicPoison_original_Dataset/hotpotqa/corpus.jsonl). Written under
    datasets/graphrag_under_fire/ rather than grag-data/input/ directly, matching
    where the other datasets' raw corpus.jsonl lives before _ensure_dataset_synced()
    (main.py) copies it into grag-data/input/ and syncs it into GraphRAG's own
    input format - this dataset isn't part of the logicPoison_original_Dataset
    submodule, so _ensure_dataset_synced() falls back to datasets/<dataset>/
    when attack_config.data_root doesn't have it.
    """
    with open(para_path, encoding="utf-8") as f:
        paragraphs = [line.strip() for line in f if line.strip()]

    records = [
        {"_id": str(i), "title": f"para_{i}", "text": p}
        for i, p in enumerate(paragraphs)
    ]
    _write_jsonl(records, clean_output)
    print(f"[OK] Ingested {len(records)} clean paragraphs -> {clean_output}")


def select_poison_texts(attack: dict, attack_type: str, n: int, seed: int = 0) -> list[str]:
    """Picks up to `n` adversarial texts of one attack entry's `attack_type`
    ("direct" -> direct_adv_texts, "indirect" -> indirect_adv_texts,
    "enhanced" -> enhanced_texts - the paper's three distinct attack
    strategies, not interchangeable variants of the same text).
    """
    key = "enhanced_texts" if attack_type == "enhanced" else f"{attack_type}_adv_texts"
    texts = attack.get(key, [])
    if not texts:
        return []
    return random.Random(seed).sample(texts, min(n, len(texts)))


def build_graphrag_under_fire_dataset(
    n_per_attack: int = 1,
    seed: int = 0,
    para_path: str | Path = "datasets/graphragUnderFire/need_save_para.txt",
    attacks_path: str | Path = "results/poisoned_data/graphrag_under_fire/test0_corpus.json",
    clean_output: str | Path = "datasets/graphrag_under_fire/corpus.jsonl",
    poisoned_dir: str | Path = "results/poisoned_data/graphrag_under_fire",
) -> None:
    """Builds the graphragUnderFire dataset: one clean corpus.jsonl (the 208
    real paragraphs, indexed by GraphRAG as usual) plus one poisoned
    corpus_<attack_type>.jsonl per attack type (direct/indirect/enhanced),
    each containing ONLY that type's inserted adversarial documents - no
    duplicated clean paragraphs, no placeholder rows, since insertions are
    scored via detect_insertions()/score() alone (original_text is never read,
    so there's nothing for a matched "original" paragraph to be used for).

    Kept as three separate poisoned files, not one mixed file, because direct/
    indirect/enhanced are different attack strategies from the paper - lumping
    them together would hide which strategy a detector actually struggles with.
    """
    with open(attacks_path, encoding="utf-8") as f:
        attacks = json.load(f)

    ingest_graphrag_under_fire_corpus(para_path, clean_output)

    poisoned_dir = Path(poisoned_dir)
    for attack_type in ATTACK_TYPES:
        poison_records = []
        for attack in attacks:
            for text in select_poison_texts(attack, attack_type, n=n_per_attack, seed=seed):
                idx = len(poison_records)
                poison_records.append({"_id": str(idx), "title": f"{attack_type}_{idx}", "text": text})

        out_path = poisoned_dir / f"corpus_{attack_type}.jsonl"
        _write_jsonl(poison_records, out_path)
        print(f"[OK] {len(poison_records)} '{attack_type}' poison docs -> {out_path}")


def main():
    # Paths
    graphrag_input = "graphrag-api/graphrag/input"
    dataset_output = "datasets/graphrag"
    
    # Create output directory
    os.makedirs(dataset_output, exist_ok=True)
    
    # Find all text files in GraphRAG input
    text_files = []
    if os.path.isdir(graphrag_input):
        text_files = [
            os.path.join(graphrag_input, f) 
            for f in os.listdir(graphrag_input) 
            if f.endswith(".txt")
        ]
    
    if not text_files:
        print(f"[ERROR] No .txt files found in {graphrag_input}")
        return
    
    print(f"[INFO] Found {len(text_files)} text files")
    
    # Create corpus and queries
    corpus_path = os.path.join(dataset_output, "corpus.jsonl")
    queries_path = os.path.join(dataset_output, "queries.jsonl")
    
    create_corpus_jsonl(text_files, corpus_path)
    create_queries_jsonl(queries_path)
    
    print(f"\n[INFO] Dataset created at: {dataset_output}")
    print(f"       - corpus.jsonl: {corpus_path}")
    print(f"       - queries.jsonl: {queries_path}")


if __name__ == "__main__":
    main()
