import sys
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random

from classes.ServerOllama import *
from classes.LLMUser import *

LLM_name = "gpt-oss:20b"

# -----------------------------
# Main Script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Multi-Dataset QA Wikipedia Retrieval Pipeline (Test Splits Only)...")

    # -----------------------------
    # Load datasets
    # -----------------------------
    datasets_dict = {
        "ARC-Challenge": load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"],
        "SciQ": load_dataset("allenai/sciq")["test"],
        "GPQA": load_dataset("Idavidrein/gpqa", "gpqa_main")["test"]
    }

    # -----------------------------
    # Initialize LLM and retriever
    # -----------------------------
    server = OllamaServer()
    llm = OllamaChat(server=server, model=LLM_name)
    retriever = LLMUser(llm=llm)

    # -----------------------------
    # Process each dataset
    # -----------------------------
    for dataset_name, subset in datasets_dict.items():
        print(f"\n📂 Processing dataset: {dataset_name} (test split, {len(subset)} examples)")

        # Load preprocessed facts
        facts_file = Path(f"preprocessed_fact/{dataset_name.lower()}_preprocessed_fact.json")
        if facts_file.exists():
            with open(facts_file, "r", encoding="utf-8") as f:
                facts_data = json.load(f)
            print(f"✅ Loaded preprocessed facts: {facts_file}")
        else:
            facts_data = {}
            print(f"⚠️ No preprocessed facts found for {dataset_name}")

        output_data = []

        for i, example in enumerate(tqdm(subset, desc=f"Processing {dataset_name}")):
            # -----------------------------
            # Extract question and choices
            # -----------------------------
            question = example.get("question", "")
            explanation = example.get("explanation")  # Default None

            if dataset_name == "SciQ":
                choices = [
                    example["correct_answer"],
                    example["distractor1"],
                    example["distractor2"],
                    example["distractor3"]
                ]
                answer_key = example["correct_answer"]
                random.shuffle(choices)  # Shuffle choices for robustness

            elif dataset_name == "ARC-Challenge":
                choices = example.get("choices", [])
                answer_key = example.get("answerKey")
                explanation = example.get("support")  # Optional

            elif dataset_name == "GPQA":
                choices = example.get("choices", [])
                answer_key = example.get("correct_answer")
                explanation = None

            else:
                choices = []
                answer_key = None

            # -----------------------------
            # Retrieve Wikipedia pages
            # -----------------------------
            try:
                wikipedia_pages = retriever.get_candidate_pages(question, choices, max_pages=5)
            except Exception as e:
                print(f"❌ Retrieval failed for Q{i}: {e}")
                wikipedia_pages = []

            # -----------------------------
            # Load choice facts if available
            # -----------------------------
            choice_facts = facts_data.get(str(i), {})

            # -----------------------------
            # Build record
            # -----------------------------
            record = {
                "dataset": dataset_name,
                "id": i,
                "question": question,
                "choices": choices,
                "answerKey": answer_key,
                "explanation": explanation,
                "retrieved_pages": wikipedia_pages,
                "choice_facts": choice_facts,
            }

            output_data.append(record)

        # -----------------------------
        # Save dataset output
        # -----------------------------
        output_path = Path(f"dataset/{dataset_name.lower()}_test_with_wiki_ref.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved {dataset_name} test data: {output_path.resolve()}")
