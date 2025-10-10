import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pprint
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from classes.LLM import LLM
from classes.LLMUser import LLMUser
from classes.AF import ArgumentationGraph

# --- Ensure project root is in sys.path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_load_json(file_path):
    """Load JSON while ignoring trailing commas and minor formatting issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r",(\s*[\]}])", r"\1", text)
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ Failed to load {file_path}: {e}")
        return []

def run_model(model_name, ranked_evidence, dataset):
    """Run the full pipeline for a given model and save results."""
    print(f"\n🚀 Running for model: {model_name}")
    output_dir = os.path.join("results", model_name.replace(":", "_"))
    os.makedirs(output_dir, exist_ok=True)

    llm_model = LLM(model=model_name)
    user = LLMUser(llm_model)
    graph_builder = ArgumentationGraph()

    combined_data = []
    y_true = []
    y_pred = []
    skipped_missing_data = 0

    for i in range(len(ranked_evidence)):
        # --- Build text from ranked evidence ---
        text_parts = []
        for j in ranked_evidence[i].keys():
            entry = ranked_evidence[i][j]
            if entry and isinstance(entry, list) and "text" in entry[0]:
                text_parts.append(entry[0]["text"])
        text = " ".join(text_parts).strip()

        # --- Collect hypotheses ---
        hypotheses = []
        facts_per_choice = dataset[i].get("facts_per_choice", {}) if i < len(dataset) else {}
        for _, fact_text in list(facts_per_choice.items())[:4]:
            if fact_text and fact_text.strip():
                hypotheses.append(fact_text.strip())

        correct_answer = dataset[i].get("correct_answer")
        possible_answers = dataset[i].get("choices")

        if not possible_answers or correct_answer is None:
            skipped_missing_data += 1
            continue

        print("\n=== Hypotheses ===")
        pprint.pprint(hypotheses)
        print("Correct answer:", correct_answer)

        # --- Build argumentation graph ---
        graph_result = graph_builder.build_from_text(
            text,
            user,
            extra_arguments=hypotheses,
            insert_at_start=True
        )

        strengths = graph_result.get("strengths", {})

        # --- Map strengths from node_id → hypothesis text ---
        strengths_text = {
            graph_builder.get_text_from_id(nid): val
            for nid, val in strengths.items()
            if graph_builder.get_text_from_id(nid) in hypotheses
        }

        if not strengths_text:
            skipped_missing_data += 1
            continue

        # --- Predict hypothesis with max strength ---
        predicted_answer = max(strengths_text, key=strengths_text.get)

        # --- Map correct_answer index to text if necessary ---
        if isinstance(correct_answer, int):
            if correct_answer < len(possible_answers):
                correct_value = possible_answers[correct_answer]
            else:
                skipped_missing_data += 1
                continue
        else:
            correct_value = correct_answer

        print("corretta ✅" if predicted_answer.lower() == correct_value.lower() else "sbagliata ❌")

        # --- Collect results ---
        y_true.append(correct_value)
        y_pred.append(predicted_answer)
        combined_data.append({
            "text": text,
            "hypotheses": hypotheses,
            "correct_answer": correct_value,
            "predicted_answer": predicted_answer,
            "strengths": strengths_text
        })

        # --- Running accuracy ---
        current_acc = accuracy_score(y_true, y_pred)
        print(f"📈 Running accuracy after {len(y_true)} samples: {current_acc:.2%}")

    print(f"\nTotal entries processed: {len(combined_data)}")
    print(f"Total entries skipped: {skipped_missing_data}")

    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        print(f"\n🎯 Final Accuracy: {accuracy:.2%}")
        print(f"📊 Final F1-score: {f1:.2%}")

        # --- Confusion matrix ---
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title(f"Confusion Matrix ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

        # --- Save outputs ---
        with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        with open(os.path.join(output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Total processed: {len(combined_data)}\n")
            f.write(f"Total skipped: {skipped_missing_data}\n")
            f.write(f"Labels: {labels}\n")

        print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ranked_evidence = safe_load_json(os.path.join(base_dir, "ranked_evidence.json"))
    dataset = safe_load_json(os.path.join(base_dir, "sciq_facts_output.json"))

    models_to_run = [
        "gemma3:12b",
        "gemma3:27b"
    ]

    for model_name in models_to_run:
        run_model(model_name, ranked_evidence, dataset)
