import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from dataset_preparer import SciQDatasetPreparer

# -----------------------------
# Output directory
# -----------------------------
output_dir = Path("preprocessed_fact")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Dataset configurations
# -----------------------------
datasets_config = {
    "SciQ": {
        "hf_name": "allenai/sciq",
        "split": "test",
        "question_key": "question",
        "answer_key": "correct_answer",
        "distractors": ["distractor1", "distractor2", "distractor3"],
        "support_key": "support",
    },
    "ARC-Challenge": {
        "hf_name": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "question_key": "question",
        "answer_key": "answerKey",
        "choices_key": "choices",
    },
    "GPQA": {
        "hf_name": "Idavidrein/gpqa",
        "config": "gpqa_main",
        "split": "test",
        "question_key": "question",
        "answer_key": "correct_answer",
        "choices_key": "choices",
    },
}

# -----------------------------
# Generate factual JSONs
# -----------------------------
for dataset_name, cfg in datasets_config.items():
    print(f"\n📘 Generating facts for {dataset_name}...")

    # Load dataset
    if "config" in cfg:
        dataset = load_dataset(cfg["hf_name"], cfg["config"])[cfg["split"]]
    else:
        dataset = load_dataset(cfg["hf_name"])[cfg["split"]]

    facts_dict = {}

    for i, example in enumerate(tqdm(dataset, desc=f"{dataset_name}")):
        q = example.get(cfg["question_key"], "")

        # Determine answer choices
        if dataset_name == "SciQ":
            choices = [
                example["correct_answer"],
                example["distractor1"],
                example["distractor2"],
                example["distractor3"]
            ]
        elif dataset_name == "ARC-Challenge":
            choices = example.get("choices", {}).get("text", [])
        elif dataset_name == "GPQA":
            choices = example.get("choices", [])
        else:
            choices = []

        # Generate “facts per choice” (placeholder or LLM-generated later)
        choice_facts = {}
        for c in choices:
            choice_facts[c] = f"Fact justification for '{c}' given the question '{q}'"

        facts_dict[str(i)] = choice_facts

    # Save the factualized file
    output_path = output_dir / f"{dataset_name.lower()}_preprocessed_fact.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {output_path.resolve()}")
