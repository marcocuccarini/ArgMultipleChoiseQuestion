# ======================================================
# file: dataset_creation/WikipediaExtraction.py
# ======================================================

import json
import os
import time
import wikipedia
from requests.exceptions import RequestException
import re

from classes.utils import load_question_file, fetch_wikipedia_pages

# ======================================================
# MAIN
# ======================================================

def main():
    input_file = "dataset/sciq_with_wiki_ref.json"
    output_file = "dataset/wiki_pages_content.json"

    # Load SciQ dataset with Wikipedia references
    data = load_question_file(input_file)
    if not data:
        print("⚠️ No data loaded from input file.")
        return

    print(f"📚 Loaded {len(data)} questions from '{input_file}'")

    page_content = []

    for i, item in enumerate(data):
        try:
            question_id = item.get("id", i)
            question = item.get("question", "")
            correct_answer = item.get("correct_answer", "")
            choices = item.get("choices", [])
            explanation = item.get("explanation", "")
            retrieved_pages = item.get("retrieved_pages", [])
            choice_facts = item.get("choice_facts", {})

            print(f"\n=== [{i+1}/{len(data)}] Processing Question ID: {question_id} ===")
            print(f"❓ Question: {question}")
            print(f"   Choices: {choices}")
            print(f"   Retrieved pages: {retrieved_pages}")

            if not retrieved_pages:
                print("⚠️ No retrieved pages for this question — skipping.")
                continue

            # Fetch Wikipedia content for all retrieved pages
            sorted_pages = sorted(retrieved_pages, key=lambda x: x[1], reverse=True)
            wiki_content = fetch_wikipedia_pages(sorted_pages)

            page_entry = {
                "id": question_id,
                "question": question,
                "correct_answer": correct_answer,
                "choices": choices,
                "choice_facts": choice_facts,     # ✅ Add choice facts
                "explanation": explanation,
                "retrieved_page": wiki_content,
                "retrieved_page_name": retrieved_pages
            }

            page_content.append(page_entry)

            # Sleep a bit to avoid hitting Wikipedia too fast
            time.sleep(0.5)

        except Exception as e:
            print(f"❌ Error processing question {i} ({question}): {e}")
            continue

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(page_content, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All Wikipedia pages saved in '{output_file}'")


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()
