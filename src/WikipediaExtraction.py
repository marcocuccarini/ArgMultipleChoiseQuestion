# file: dataset_creation/WikipediaExtraction.py

import json
import wikipedia
import os
import time
from requests.exceptions import RequestException
import re

from classes.utils import *

# ----------------------------
# Main
# ----------------------------
def main():
    input_file = "dataset/sciq_with_wiki_ref.json"
    output_file = "dataset/wiki_pages_content.json"

    data = load_question_file(input_file)
    if not data:
        return
    '''with open(input_file, 'r', encoding='utf-8') as f:
        data=json.load(f)'''
        
    retrieved_pages = list([item["retrieved_pages"] for item in data])
    questions = list([item["question"] for item in data])
    ids = list([item["id"] for item in data])
    correct_answers =list([item["correct_answer"] for item in data])
    choiceses =list([item["choices"] for item in data])
    explanations = list([item["explanation"] for item in data])
    
    page_content=[]

    for i, title in enumerate(retrieved_pages):
        
        page_content.append({
            "id": ids[i],
            "question": questions[i],
            "correct_answer": correct_answers[i],
            "choices": choiceses[i],
            "explanation": explanations[i],
            "retrieved_page": fetch_wikipedia_pages(sorted(title, key=lambda x: x[1])),
}
            )


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(page_content, f, ensure_ascii=False, indent=2)
    print(f"✅ Tutte le pagine Wikipedia salvate in '{output_file}'")

if __name__ == "__main__":
    main()
