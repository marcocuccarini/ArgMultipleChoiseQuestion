import sys
import os
import json
from datasets import load_dataset

from classes.ServerOllama import *
from classes.LLMUser import *


LLM_name="gpt-oss:20b"

# -----------------------------
# Main Script
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting SciQ + Wikipedia Retrieval Pipeline...")

    # Load SciQ dataset
    dataset = load_dataset("allenai/sciq")

    # Load preprocessed facts
    with open("preprocessed_fact/facts_only.json", "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # Initialize Ollama LLM and retriever
    server = OllamaServer()
    llm = OllamaChat(server=server, model=LLM_name)
    retriever = LLMUser(llm=llm)

    output_data = []

    subset = dataset["test"]  # ✅ Correct dataset iteration

    # Process subset of SciQ dataset
    for i, example in enumerate(subset):
        question = example["question"]
        choices = [
            example["correct_answer"],
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
        ]

        print(f"\n🔍 Processing Q{i}: {question[:60]}...")

        # Retrieve Wikipedia pages
        wikipedia_pages = retriever.get_candidate_pages(question, choices, max_pages=5)

        # Get facts if available
        choice_facts = facts_data[i] if i < len(facts_data) else {}

        # Build record
        record = {
            "id": i,
            "question": question,
            "choices": choices,
            "correct_answer": example["correct_answer"],
            "explanation": example["support"],
            "retrieved_pages": wikipedia_pages,
            "choice_facts": choice_facts,
        }

        output_data.append(record)

    # Save combined data
    output_path = Path("dataset/sciq_with_wiki_ref.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved: {output_path.resolve()}")