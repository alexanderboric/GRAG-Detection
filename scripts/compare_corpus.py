#!/usr/bin/env python3
"""
Compare original and poisoned corpus to find differences (entity swaps).
"""

import json
import sys
from pathlib import Path
from difflib import unified_diff, SequenceMatcher
from typing import List, Dict, Tuple

def load_jsonl(filepath):
    """Load JSONL file and return list of JSON objects."""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    return records

def extract_diffs(original_records: List[Dict], poisoned_records: List[Dict]) -> List[Dict]:
    """
    Extract all differences between original and poisoned corpus.
    Returns a list of diff objects with detailed information.
    """
    diffs = []
    
    for i, (orig, pois) in enumerate(zip(original_records, poisoned_records)):
        orig_text = orig.get('text', '')
        pois_text = pois.get('text', '')
        
        if orig_text != pois_text:
            # Calculate similarity ratio
            matcher = SequenceMatcher(None, orig_text, pois_text)
            similarity = matcher.ratio()
            
            # Find changed portions
            changed_portions = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != 'equal':
                    changed_portions.append({
                        'operation': tag,  # 'replace', 'delete', 'insert'
                        'original': orig_text[i1:i2],
                        'poisoned': pois_text[j1:j2],
                        'original_pos': (i1, i2),
                        'poisoned_pos': (j1, j2)
                    })
            
            diff_obj = {
                'record_id': i,
                'original_record': orig,
                'poisoned_record': pois,
                'original_text': orig_text,
                'new_text': pois_text,
                'similarity': similarity,
                'num_changes': len(changed_portions),
                'changes': changed_portions
            }
            diffs.append(diff_obj)
    
    return diffs

def extract_unchanged(original_records: List[Dict], poisoned_records: List[Dict]) -> List[Dict]:
    """
    Records left untouched by the poisoning attack, shaped like extract_diffs()
    output so they can serve as the clean/negative baseline for detector benchmarks.
    """
    unchanged = []

    for i, (orig, pois) in enumerate(zip(original_records, poisoned_records)):
        text = orig.get('text', '')
        if text == pois.get('text', ''):
            unchanged.append({
                'record_id': i,
                'original_record': orig,
                'poisoned_record': pois,
                'original_text': text,
                'new_text': text,
                'similarity': 1.0,
                'num_changes': 0,
                'changes': []
            })

    return unchanged

def verify_diff(diff: Dict) -> bool:
    """
    can be used to verify poisoning.
    Returns True if diff is valid, False otherwise.
    """
    required_keys = ['original_text', 'new_text', 'similarity', 'changes']
    if not all(key in diff for key in required_keys):
        return False
    
    if diff['original_text'] == diff['new_text']:
        return False  # No actual difference
    
    return True

def filter_diffs_by_similarity(diffs: List[Dict], min_similarity: float = 0.0, max_similarity: float = 1.0) -> List[Dict]:
    """Filter diffs by similarity threshold."""
    return [d for d in diffs if min_similarity <= d['similarity'] <= max_similarity]

def filter_diffs_by_num_changes(diffs: List[Dict], min_changes: int = 1, max_changes: int = None) -> List[Dict]:
    """Filter diffs by number of changes."""
    filtered = [d for d in diffs if d['num_changes'] >= min_changes]
    if max_changes is not None:
        filtered = [d for d in filtered if d['num_changes'] <= max_changes]
    return filtered

def print_diff_summary(diffs: List[Dict]) -> None:
    """Print a summary of all diffs."""
    if not diffs:
        print("No differences found.")
        return
    
    print(f"Total diffs found: {len(diffs)}\n")
    
    similarities = [d['similarity'] for d in diffs]
    changes_counts = [d['num_changes'] for d in diffs]
    
    print(f"Similarity statistics:")
    print(f"  Min: {min(similarities):.3f}")
    print(f"  Max: {max(similarities):.3f}")
    print(f"  Avg: {sum(similarities) / len(similarities):.3f}")
    print(f"\nChanges per record:")
    print(f"  Min: {min(changes_counts)}")
    print(f"  Max: {max(changes_counts)}")
    print(f"  Avg: {sum(changes_counts) / len(changes_counts):.1f}")

def print_diff_details(diff: Dict, text_preview_len: int = 300) -> None:
    """Print detailed information about a single diff."""
    print(f"Record #{diff['record_id']}")
    print(f"Similarity: {diff['similarity']:.3f}")
    print(f"Number of changes: {diff['num_changes']}\n")
    
    print("ORIGINAL:")
    print("-" * 80)
    text = diff['original_text']
    print(text[:text_preview_len] + ("..." if len(text) > text_preview_len else ""))
    
    print("\nPOISONED:")
    print("-" * 80)
    text = diff['new_text']
    print(text[:text_preview_len] + ("..." if len(text) > text_preview_len else ""))
    
    print("\nCHANGES:")
    print("-" * 80)
    for idx, change in enumerate(diff['changes'][:5], 1):  # Show first 5 changes
        print(f"Change #{idx} ({change['operation']}):")
        print(f"  Original: {change['original'][:100]}")
        print(f"  Poisoned: {change['poisoned'][:100]}")
    
    if len(diff['changes']) > 5:
        print(f"  ... and {len(diff['changes']) - 5} more changes")

def export_diffs_to_json(diffs: List[Dict], output_file: str) -> None:
    """Export diffs to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(diffs, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(diffs)} diffs to {output_file}")

def find_diffs_interactive(original_file, poisoned_file, num_examples=None):

    """Compare original and poisoned corpus and show diffs interactively."""
    
    original = load_jsonl(original_file)
    poisoned = load_jsonl(poisoned_file)
    
    if not original or not poisoned:
        print("Error loading files")
        return []
    
    print(f"Original corpus: {len(original)} records")
    print(f"Poisoned corpus: {len(poisoned)} records")
    print("\n" + "="*80 + "\n")
    
    # Extract all diffs
    diffs = extract_diffs(original, poisoned)
    
    # Print summary
    print_diff_summary(diffs)
    print("\n" + "="*80 + "\n")
    
    # Show examples
    num_to_show = num_examples if num_examples else min(5, len(diffs))
    for i, diff in enumerate(diffs[:num_to_show], 1):
        print(f"EXAMPLE #{i}:\n")
        print_diff_details(diff)
        print("\n" + "="*80 + "\n")
    
    if len(diffs) > num_to_show:
        print(f"Showing {num_to_show} of {len(diffs)} diffs")
    
    return diffs
# method to a applie a single diff to the 
def apply_diff():
    #regex find and replace of a the original with the poisoned, then detection
    print("applying diff") 
    
if __name__ == "__main__":
    original_file = "datasets/logicPoison_original_Dataset/hotpotqa/corpus.jsonl"
    poisoned_file = "results/poisoned_data/hotpotqa/corpus.jsonl"
    outputdir = poisoned_file.split("/corpus.jsonl")[0] 
    
    # Get all diffs
    diffs = find_diffs_interactive(original_file, poisoned_file, num_examples=3)
    
    # Optional: Export to JSON for further analysis
    if diffs:
        export_diffs_to_json(diffs, outputdir+"diffs_analysis.json")
        
        # Example: Filter diffs by similarity
        print("\n" + "="*80)
        print("FILTERED ANALYSIS (Similarity < 0.5):")
        print("="*80 + "\n")
        low_sim_diffs = filter_diffs_by_similarity(diffs, max_similarity=0.5)
        print(f"Found {len(low_sim_diffs)} highly poisoned records (similarity < 0.5)")
        # printing a document, with very low similarity 
        
        if low_sim_diffs:
            print("low similarity example:")
            print_diff_details(low_sim_diffs[0])
        
        # Example: Verify each diff is valid
        print("\n" + "="*80)
        print("DIFF VALIDITY CHECK:")
        print("="*80 + "\n")
        valid_diffs = [d for d in diffs if verify_diff(d)]
        print(f"Valid diffs: {len(valid_diffs)} / {len(diffs)}")

