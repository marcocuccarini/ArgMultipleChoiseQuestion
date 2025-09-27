import sys
import os
import json
from datasets import load_dataset

# Add classes folder to module path
sys.path.append(os.path.join(os.path.dirname(__file__), "../classes"))

from LLM import *
from LLM_IR_wiki import *


# -----------------------------
# Save SciQ + Wikipedia retrievals + explanations + facts to JSON
# -----------------------------
if __name__ == "__main__":

    # Load SciQ dataset
    dataset = load_dataset("allenai/sciq")

    # Load external facts JSON
    with open("preprocessed_fact/facts_only.json", "r", encoding="utf-8") as f:
        facts_data = json.load(f)

    # Initialize your LLM wrapper
    llm = LLM(model="gemma3:1b")

    # Initialize retriever
    retriever = LLMRetrieverPagesOnly(llm=llm)

    output_data = []

    # Loop over first N examples (adjust as needed)
    for i, example in enumerate(dataset["validation"][:50]):  # change slice to control size
        question = example["question"]
        choices = [
            example["correct_answer"],
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
        ]

        # Retrieve Wikipedia pages
        wikipedia_pages = retriever.get_candidate_pages(question, choices, max_pages=5)

        # Get facts if available (make sure index is within bounds)
        choice_facts = facts_data[i] if i < len(facts_data) else {}

        # Store in dictionary
        record = {
            "id": i,
            "question": question,
            "choices": choices,
            "correct_answer": example["correct_answer"],
            "explanation": example["support"],
            "retrieved_pages": wikipedia_pages,
            "choice_facts": choice_facts,  # ✅ directly include facts from file
        }

        output_data.append(record)

    # Save to JSON file
    with open("sciq_with_wiki.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✅ JSON file saved as sciq_with_wiki.json")
