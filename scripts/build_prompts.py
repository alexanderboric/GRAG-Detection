"""Generates prompts.jsonl: for each scenario (llm_as_a_judge / grag_as_a_judge x insertion /
edit), asks an LLM (via the project's usual OPENAI_BASE_URL/OPENAI_API_KEY client) for N different
judging-prompt variants in one call, and writes the results out with an id per prompt so they can
be referenced (e.g. by Optuna as a categorical param) at detection time.

Run directly (`python scripts/build_prompts.py`) to (re)generate prompts.jsonl - overwrites the
file each time.

WARNING: overwriting drops the hand-written grag_as_a_judge variants v10+ that
scripts/add_gaaj_variants.py appends, and renumbers everything else, invalidating the `variant`
indices cached in results/detector_best_params/*.json. Re-run add_gaaj_variants.py afterwards and
re-optimize.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OUT_PATH = Path(__file__).parent.parent / "prompts.jsonl"
MODEL = "mistral-small-4"
N_VARIANTS = 10

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
    base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
)

SCENARIOS = [
    ("llm_as_a_judge", "insertion"),
    ("llm_as_a_judge", "edit"),
    ("grag_as_a_judge", "insertion"),
    ("grag_as_a_judge", "edit"),
]

META_PROMPT = """You are designing {n} different system prompts for an LLM that judges whether a {target} \
has been factually "poisoned" - deliberately altered to contain incorrect or misleading information \
(swapped named entities, wrong dates/numbers, contradicted relationships, implausible claims) even \
though it still reads fluently. This is NOT about grammar or writing quality.

{mode_desc}
{grag_desc}

Give me {n} genuinely different prompt-engineering strategies, for example (vary freely, invent your \
own too): a plain direct baseline, an explicit numeric rubric with anchors at low/mid/high severity, \
chain-of-thought reasoning before the final score, few-shot worked examples, a checklist that scores \
entities/dates/relationships/logic separately, a probability estimate instead of an integer, an \
adversarial self-critique that argues both sides before deciding, a neutral fact-checker framing \
that avoids "poisoning"/security language, a topic/title-grounded prompt that leans on background \
knowledge, and an entity-by-entity enumeration approach.

Requirements for EVERY variant's system_prompt:
- Self-contained (a judge model sees nothing but this text plus the {target} itself).
- Must end with a precise, unambiguous instruction for how to format the reply, matching whichever \
output_format you choose for that variant: "int_0_10" -> reply with ONLY a single integer 0-10, \
"int_0_10_cot" -> reason briefly, then end with a final line "SCORE: <n>" (n = integer 0-10), \
"prob_0_1" -> reply with ONLY a decimal probability between 0.00 and 1.00.
{context_requirement}

Use plain ASCII punctuation only - a hyphen "-", not an em/en dash or any other unicode punctuation.

Reply with ONLY a JSON object of this exact shape, no other text:
{{"prompts": [{{"strategy": "<short_snake_case_name>", "output_format": "int_0_10|int_0_10_cot|prob_0_1", "system_prompt": "<full prompt text>"}}, ...]}}"""

MODE_DESC = {
    "insertion": "You are judging a single document in isolation (no original to compare against) - decide whether it looks internally altered/implausible on its own.",
    "edit": "You are comparing an ORIGINAL document against a NEW (possibly edited) version - decide whether the edit introduced factual divergence from the original.",
}
GRAG_DESC = (
    'Each variant will also be given retrieved CONTEXT from a knowledge graph built on this corpus, '
    'reflecting established facts/relationships among entities mentioned in the {target}. Every '
    'system_prompt MUST include the literal placeholder "{{context}}" (unmodified, to be filled in '
    'later) at the point where that retrieved context should appear, and should instruct the judge '
    'to treat it as a reference - contradicting or unsupported-by-context content is more suspicious, '
    'but the context may be incomplete, so mere absence is not itself evidence of alteration.'
)
CONTEXT_REQUIREMENT = (
    '- Must include the literal placeholder "{context}" exactly once, at the point where retrieved '
    "context should be inserted."
)


def generate_variants(detector: str, mode: str) -> list[dict]:
    target = "document" if mode == "insertion" else "edit"
    is_grag = detector == "grag_as_a_judge"
    meta_prompt = META_PROMPT.format(
        n=N_VARIANTS,
        target=target,
        mode_desc=MODE_DESC[mode],
        grag_desc=GRAG_DESC.format(target=target) if is_grag else "",
        context_requirement=CONTEXT_REQUIREMENT if is_grag else "",
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": meta_prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    variants = data["prompts"]
    for v in variants:
        # Some models occasionally emit a literal U+FFFD replacement char in place of an
        # en/em dash despite the ASCII-only instruction above - clean up defensively.
        v["system_prompt"] = v["system_prompt"].replace("�", "-")
    if is_grag:
        variants = [v for v in variants if "{context}" in v["system_prompt"]]
        missing = len(data["prompts"]) - len(variants)
        if missing:
            print(f"  WARNING: dropped {missing} {detector}/{mode} variant(s) missing the {{context}} placeholder")
    return variants


def main() -> None:
    rows = []
    for detector, mode in SCENARIOS:
        print(f"[{detector}/{mode}] requesting {N_VARIANTS} variants...")
        variants = generate_variants(detector, mode)
        print(f"  got {len(variants)} usable variants")
        for i, v in enumerate(variants):
            rows.append({
                "id": f"{detector}_{mode}_{v['strategy']}",
                "detector": detector,
                "mode": mode,
                # `variant` (not `strategy`) is the cross-mode pairing key: insertion and edit are
                # drafted in separate LLM calls, so the model's own strategy *names* don't line up
                # between modes (e.g. "numeric_rubric_anchors" vs "numeric_rubric_with_anchors") -
                # position in the requested list is the only reliable correspondence.
                "variant": i,
                "strategy": v["strategy"],
                "output_format": v.get("output_format", "int_0_10"),
                "system_prompt": v["system_prompt"],
            })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
