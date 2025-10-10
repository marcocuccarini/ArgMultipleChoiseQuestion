# ======================================================
# MAIN SCRIPT: GRAPH-BASED QA PREDICTION
# ======================================================
import os
import json
import networkx as nx
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms
from classes.LLM import LLM
from classes.LLMUser import LLMUser
from classes.AF import *
# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "gpt-oss:20b"  # change to your LLM model
DATASET_FILE = "dataset/sciq_top20_intervals_with_facts.json"
RANKED_EVIDENCE_FILE = "dataset/ranked_evidence.json"

# ----------------------------
# Load dataset and evidence
# ----------------------------
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

with open(RANKED_EVIDENCE_FILE, "r", encoding="utf-8") as f:
    ranked_evidence = json.load(f)

# ----------------------------
# Initialize LLM, User, Graph
# ----------------------------
llm_model = LLM(model=MODEL_NAME)
user = LLMUser(llm_model)
graph_builder = ArgumentationGraph()

output_dir = os.path.join("results", MODEL_NAME.replace(":", "_"))
os.makedirs(output_dir, exist_ok=True)

y_true = []
y_pred = []
combined_data = []

# ----------------------------
# Process each question
# ----------------------------
for i, entry in enumerate(ranked_evidence):
    text_parts = []
    for key in entry:
        items = entry[key]
        if items and isinstance(items, list) and "text" in items[0]:
            text_parts.append(items[0]["text"])
    text = " ".join(text_parts).strip()

    facts_per_choice = dataset[i].get("choice_facts", {}) if i < len(dataset) else {}
    hypotheses = [fact.strip() for fact in facts_per_choice.values() if fact.strip()]

    if not hypotheses or not text:
        continue

    correct_answer = dataset[i].get("correct_answer")
    possible_answers = dataset[i].get("choices")

    # --- Build argumentation graph ---
    graph_result = graph_builder.build_from_text(
        text=text,
        llm_user=user,
        extra_arguments=hypotheses,
        insert_at_start=True
    )

    # --- Map node strengths to hypothesis text ---
    strengths = graph_result.get("strengths", {})
    strengths_text = {}
    for nid, arg_strength in strengths.items():
        node_text = graph_builder.get_text_from_id(nid)
        if node_text in hypotheses:
            strengths_text[node_text] = arg_strength

    if not strengths_text:
        continue

    # --- Prediction ---
    predicted_answer = max(strengths_text, key=strengths_text.get)

    if isinstance(correct_answer, int) and correct_answer < len(possible_answers):
        correct_value = possible_answers[correct_answer]
    else:
        correct_value = correct_answer

    y_true.append(correct_value)
    y_pred.append(predicted_answer)

    combined_data.append({
        "text": text,
        "hypotheses": hypotheses,
        "correct_answer": correct_value,
        "predicted_answer": predicted_answer,
        "strengths": strengths_text
    })

    # --- PRINT PREDICTION ---
    print("\n----------------------------------------")
    print(f"🧩 Question {i+1}")
    print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
    print("Hypotheses & Strengths:")
    for hyp, val in strengths_text.items():
        print(f"  - {hyp[:80]}{'...' if len(hyp) > 80 else ''}: {val:.3f}")
    print(f"✅ Correct answer:   {correct_value}")
    print(f"🔮 Predicted answer: {predicted_answer}")
    print("----------------------------------------")

# ----------------------------
# Save results
# ----------------------------
with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_dir, "y_true_y_pred.json"), "w", encoding="utf-8") as f:
    json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=2, ensure_ascii=False)

# ----------------------------
# Compute and print accuracy
# ----------------------------
correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
accuracy = correct_count / len(y_true) if y_true else 0.0

print("\n========================================")
print(f"🏁 Finished! Total evaluated: {len(y_true)}")
print(f"✅ Correct: {correct_count}")
print(f"📊 Accuracy: {accuracy*100:.2f}%")
print(f"Results saved in: {output_dir}")
print("========================================")
