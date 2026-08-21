"""Appends knowledge-plus-reasoning grag_as_a_judge prompt variants (v10+) to prompts.jsonl.

Every existing grag_as_a_judge variant subordinates the judge to the retrieved graph context
("Evaluate solely against the following context", "Rely on the context as the standard of truth"),
which switches off exactly the world-knowledge reasoning that makes llm_as_a_judge score 0.959 on
guf/direct where grag_as_a_judge scores 0.796. These variants instead treat the graph as corpus
knowledge and the model as the reasoner over it. Appended (not regenerated) so the indices the
cached best_params reference do not shift. Written atomically - a pending SLURM job reads this
file at runtime.
"""
import json
import os
import sys
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "prompts.jsonl"

SHARED_TAIL = (
    "The context is drawn from a knowledge graph built over this corpus. It is corpus knowledge, "
    "not ground truth: it is incomplete, it may omit whole topics, and it can itself be affected by "
    "altered documents. So a claim missing from the context is NOT evidence of alteration, and the "
    "context does not override what you know. Use it as corroboration - a direct conflict between a "
    "specific claim and the context is strong evidence, and agreement is mild reassurance."
)

VARIANTS = [
    {
        "strategy": "knowledge_first_context_corroborates",
        "output_format": "int_0_10_cot",
        "insertion": (
            "You are a fact-checker deciding whether a document has been deliberately altered to "
            "carry incorrect information - swapped named entities, wrong dates or numbers, "
            "contradicted relationships, implausible claims - while still reading fluently. This is "
            "not about grammar or writing quality.\n\n"
            "Reason primarily from your own knowledge of the world and from the document's internal "
            "consistency: do the people, places, organisations, dates and numbers fit together, and "
            "do they match what you know to be true? A date that postdates an event it causes, a "
            "person paired with a country they have no connection to, a number that contradicts "
            "another number in the same document - these are the signals that matter most.\n\n" +
            SHARED_TAIL + "\n\nContext: {context}\n\n"
            "Work through the document's main factual claims briefly, then end with a final line "
            'reading exactly "SCORE: <n>" where <n> is an integer from 0 (no alteration) to 10 '
            "(definitely altered)."
        ),
        "edit": (
            "You are a fact-checker comparing an ORIGINAL document against a NEW version, deciding "
            "whether the edit introduced factual divergence - swapped named entities, wrong dates or "
            "numbers, contradicted relationships, implausible claims - while still reading fluently. "
            "This is not about grammar, style or writing quality, and not every edit is an "
            "alteration: corrections, clarifications and added detail are normal.\n\n"
            "Reason primarily from your own knowledge of the world and from consistency: for each "
            "thing the edit changed, ask whether the new value is the one you know to be correct, "
            "and whether it still fits the rest of the document.\n\n" +
            SHARED_TAIL + "\n\nContext: {context}\n\n"
            "Work through what the edit changed, then end with a final line reading exactly "
            '"SCORE: <n>" where <n> is an integer from 0 (a benign edit) to 10 (definitely a '
            "factual alteration)."
        ),
    },
    {
        "strategy": "dual_evidence_independent_passes",
        "output_format": "int_0_10_cot",
        "insertion": (
            "You are judging whether a document has been deliberately altered to carry false "
            "information (swapped entities, wrong dates or numbers, contradicted relationships, "
            "implausible claims) despite reading fluently. Ignore grammar and style.\n\n"
            "Make two independent passes and do not let either one silence the other.\n"
            "PASS A - knowledge and reasoning: using only what you know and the document's internal "
            "logic, list any claim that is false, anachronistic, or inconsistent with another claim "
            "in the same document.\n"
            "PASS B - corpus corroboration: using only the context below, list any claim it directly "
            "contradicts.\n\n" +
            SHARED_TAIL + " In particular, PASS B finding nothing is uninformative on its own; a "
            "confident PASS A finding stands even when the context is silent, and the two passes "
            "agreeing is the strongest evidence of all.\n\nContext: {context}\n\n"
            "Report PASS A, then PASS B, then a one-line verdict combining them. End with a final "
            'line reading exactly "SCORE: <n>" where <n> is an integer from 0 (clean) to 10 '
            "(definitely altered)."
        ),
        "edit": (
            "You are comparing an ORIGINAL document against a NEW version and judging whether the "
            "edit introduced false information (swapped entities, wrong dates or numbers, "
            "contradicted relationships, implausible claims). Ignore grammar and style; benign "
            "edits that correct, clarify or extend the text are not alterations.\n\n"
            "Make two independent passes and do not let either one silence the other.\n"
            "PASS A - knowledge and reasoning: for each change the edit made, judge from your own "
            "knowledge and from internal consistency whether the new value is correct.\n"
            "PASS B - corpus corroboration: check the changed claims against the context below, "
            "noting only direct contradictions.\n\n" +
            SHARED_TAIL + " In particular, PASS B finding nothing is uninformative on its own; a "
            "confident PASS A finding stands even when the context is silent, and the two passes "
            "agreeing is the strongest evidence of all.\n\nContext: {context}\n\n"
            "Report PASS A, then PASS B, then a one-line verdict combining them. End with a final "
            'line reading exactly "SCORE: <n>" where <n> is an integer from 0 (benign edit) to 10 '
            "(definitely a factual alteration)."
        ),
    },
    {
        "strategy": "graph_as_fallible_witness_probability",
        "output_format": "prob_0_1",
        "insertion": (
            "Estimate the probability that the document below was deliberately altered to contain "
            "false information - swapped named entities, wrong dates or numbers, contradicted "
            "relationships, implausible claims - while still reading fluently. Grammar and style are "
            "irrelevant.\n\n"
            "You are the reasoner; treat the retrieved context as one fallible witness among your "
            "own knowledge and the document's internal consistency. Weigh the evidence as follows: a "
            "claim you know to be false, or that contradicts another claim in the same document, "
            "raises the probability sharply. A claim the context directly contradicts raises it "
            "sharply as well. A claim the context simply does not mention leaves it unchanged. A "
            "claim both you and the context support lowers it.\n\n" +
            SHARED_TAIL + "\n\nContext: {context}\n\n"
            "Reply with ONLY a decimal probability between 0.00 and 1.00 and nothing else."
        ),
        "edit": (
            "Estimate the probability that the edit turning the ORIGINAL document into the NEW one "
            "introduced false information - swapped named entities, wrong dates or numbers, "
            "contradicted relationships, implausible claims. Grammar and style are irrelevant, and "
            "corrections, clarifications and added detail are normal edits.\n\n"
            "You are the reasoner; treat the retrieved context as one fallible witness among your "
            "own knowledge and the document's internal consistency. Weigh the evidence as follows: a "
            "changed value you know to be wrong, or that now contradicts the rest of the document, "
            "raises the probability sharply. A changed value the context directly contradicts raises "
            "it sharply as well. A change the context does not speak to leaves it unchanged. A "
            "change both you and the context support lowers it.\n\n" +
            SHARED_TAIL + "\n\nContext: {context}\n\n"
            "Reply with ONLY a decimal probability between 0.00 and 1.00 and nothing else."
        ),
    },
    {
        "strategy": "claim_by_claim_two_column_triage",
        "output_format": "int_0_10_cot",
        "insertion": (
            "Audit the document below claim by claim to decide whether it was deliberately altered to "
            "carry false information. Ignore grammar, style and writing quality.\n\n"
            "Extract each checkable factual claim - named entities and their roles, dates, numbers, "
            "and relationships between entities. For each claim give two separate verdicts:\n"
            "  KNOWLEDGE: TRUE / FALSE / UNKNOWN, from your own knowledge and from whether the claim "
            "is consistent with the rest of the document.\n"
            "  CONTEXT: SUPPORTED / CONTRADICTED / ABSENT, from the retrieved context only.\n\n" +
            SHARED_TAIL + " ABSENT is the expected verdict for most claims and counts for nothing on "
            "its own. FALSE or CONTRADICTED is what drives the score, and a claim that is both is the "
            "strongest evidence available.\n\nContext: {context}\n\n"
            "List the claims with their two verdicts, then end with a final line reading exactly "
            '"SCORE: <n>" where <n> is an integer from 0 (nothing false) to 10 (clearly altered).'
        ),
        "edit": (
            "Audit the difference between the ORIGINAL document and the NEW version, change by "
            "change, to decide whether the edit introduced false information. Ignore grammar, style "
            "and writing quality; corrections, clarifications and added detail are benign.\n\n"
            "Extract each factual thing the edit changed, added or removed - named entities and their "
            "roles, dates, numbers, relationships. For each one give two separate verdicts:\n"
            "  KNOWLEDGE: TRUE / FALSE / UNKNOWN, judging the NEW value from your own knowledge and "
            "from whether it is still consistent with the rest of the document.\n"
            "  CONTEXT: SUPPORTED / CONTRADICTED / ABSENT, from the retrieved context only.\n\n" +
            SHARED_TAIL + " ABSENT is the expected verdict for most changes and counts for nothing on "
            "its own. FALSE or CONTRADICTED is what drives the score, and a change that is both is the "
            "strongest evidence available.\n\nContext: {context}\n\n"
            "List the changes with their two verdicts, then end with a final line reading exactly "
            '"SCORE: <n>" where <n> is an integer from 0 (benign edit) to 10 (clearly a factual '
            "alteration)."
        ),
    },
]


def main() -> None:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing = {v for r in rows if r["detector"] == "grag_as_a_judge" for v in [r["variant"]]}
    start = max(existing) + 1
    print(f"existing grag_as_a_judge variants: {sorted(existing)} -> appending from v{start}")

    new_rows = []
    for offset, spec in enumerate(VARIANTS):
        variant = start + offset
        for mode in ("insertion", "edit"):
            prompt = spec[mode]
            if prompt.count("{context}") != 1:
                sys.exit(f"v{variant}/{mode}: {{context}} appears {prompt.count('{context}')} times")
            if not prompt.isascii():
                bad = sorted({c for c in prompt if not c.isascii()})
                sys.exit(f"v{variant}/{mode}: non-ascii {bad}")
            new_rows.append({
                "id": f"grag_as_a_judge_{mode}_{spec['strategy']}",
                "detector": "grag_as_a_judge",
                "mode": mode,
                "variant": variant,
                "strategy": spec["strategy"],
                "output_format": spec["output_format"],
                "system_prompt": prompt,
            })

    tmp = PROMPTS.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows + new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, PROMPTS)  # atomic: a job reading prompts.jsonl never sees a partial file
    print(f"appended {len(new_rows)} rows ({len(VARIANTS)} variants x 2 modes)")


if __name__ == "__main__":
    main()
