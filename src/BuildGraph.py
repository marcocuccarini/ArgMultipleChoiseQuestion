# ======================================================
# GRAPH-BASED QA PREDICTION WITH TOP-5 INTERVALS
# ======================================================

import os
import json
import networkx as nx
from classes.LLMUser import LLMUser
from classes.AF import ArgumentationGraph
from classes.ServerOllama import OllamaServer, OllamaChat
from classes.PromptBuilder import *
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms

# ======================================================
# CONFIGURATION
# ======================================================

MODEL_NAME = "gpt-oss:20b"  # Ollama model
RETRIEVE_VALUE = 5           # Number of paragraphs retrieved at each step
DATASET_FILE = "dataset/wiki_ranked_pages.json"
OUTPUT_DIR = os.path.join("results", MODEL_NAME.replace(":", "_"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize LLM and graph handler
server = OllamaServer()
llm = OllamaChat(server, MODEL_NAME)
llm_user = LLMUser(llm)
graph_builder = ArgumentationGraph()

# Load dataset
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ======================================================
# PROCESS QUESTIONS
# ======================================================

y_true = []
y_pred = []
combined_data = []

for idx, entry in enumerate(dataset):
    print(f"\n=== Processing Question {idx + 1} ===")

    question = entry.get("question", "")
    possible_answers = entry.get("choices", [])
    correct_answer = entry.get("correct_answer", "")

    # Hypotheses = all possible answers
    hypotheses = list({choice: choice for choice in possible_answers}.values())

    predicted_answer = None
    strengths_text = {}
    graph_result = None

    # Get retrieved pages and ranked page data
    retrieved_pages = entry.get("retrieved_page_name", [])
    ranked_pages = entry.get("ranked_pages", {})

    if not retrieved_pages:
        print(f"⚠️ No retrieved pages for question {idx + 1}")
        continue

    # Iterate through pages sorted by their ranking scores
    for page_name, score in sorted(retrieved_pages, key=lambda x: x[1], reverse=True):
        page_data = ranked_pages.get(page_name, {})

        for section in ["summary_ranked", "other_ranked"]:
            paragraphs = page_data.get(section, [])

            # Extract paragraph texts only
            top_intervals = [p.get("text", "").strip() for p in paragraphs[:RETRIEVE_VALUE] if p.get("text")]
            combined_text = " ".join(top_intervals).strip()

            if not combined_text:
                continue

            print(f"📄 Using {len(combined_text.split())} words from page '{page_name}' [{section}]")

            try:
                # Build or extend the graph
                if graph_result is None:
                    graph_result = graph_builder.build_from_text(
                        text=combined_text,
                        llm_user=llm_user,
                        extra_arguments=hypotheses,
                        insert_at_start=True,
                        max_arguments=5
                    )
                else:
                    graph_result = graph_builder.extend_from_text(
                        text=combined_text,
                        llm_user=llm_user
                    )

            except Exception as e:
                print(f"❌ Error while building/extending graph for {page_name}: {e}")
                continue

            if not graph_result or "graph" not in graph_result:
                print(f"⚠️ No valid graph returned for {page_name} [{section}]")
                continue

            # Extract strengths
            strengths = graph_result.get("strengths", {})
            strengths_text = {}

            for nid, strength_val in strengths.items():
                node_text = graph_builder.get_text_from_id(nid)
                if not node_text:
                    continue
                for hyp in hypotheses:
                    if hyp.lower() in node_text.lower():
                        strengths_text[hyp] = strength_val

            # Choose the best hypothesis if confident
            if strengths_text:
                sorted_strengths = sorted(strengths_text.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_strengths) == 1 or (sorted_strengths[0][1] - sorted_strengths[1][1]) > 0.15:
                    predicted_answer = sorted_strengths[0][0]
                    break

        if predicted_answer:
            break

    # Fallback if still unknown
    if not predicted_answer and strengths_text:
        predicted_answer = max(strengths_text, key=strengths_text.get)
    elif not predicted_answer:
        predicted_answer = "Unknown"

    # Store results
    y_true.append(correct_answer)
    y_pred.append(predicted_answer)
    combined_data.append({
        "question": question,
        "hypotheses": hypotheses,
        "predicted_answer": predicted_answer,
        "correct_answer": correct_answer,
        "strengths": strengths_text,
    })

    # ======================================================
    # SAVE FINAL GRAPH FOR THIS QUESTION
    # ======================================================
    if graph_result and "graph" in graph_result:
        graph_file = os.path.join(OUTPUT_DIR, f"graph_question_{idx + 1}.json")
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(graph_result["graph"]), f, indent=2)
        print(f"📁 Graph saved to {graph_file}")
    else:
        print("⚠️ No graph available to save for this question.")

    print(f"✅ Correct: {correct_answer} 🔮 Predicted: {predicted_answer}")
    print("Hypotheses strengths:", strengths_text)

# ======================================================
# SAVE RESULTS & COMPUTE ACCURACY
# ======================================================

with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "y_true_y_pred.json"), "w", encoding="utf-8") as f:
    json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=2, ensure_ascii=False)

correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
accuracy = correct_count / len(y_true) if y_true else 0.0

print("\n========================================")
print(f"🏁 Finished! Total evaluated: {len(y_true)}")
print(f"✅ Correct: {correct_count}")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")
print(f"Results saved in: {OUTPUT_DIR}")
print("========================================")
