# ======================================================
# GRAPH-BASED QA PREDICTION WITH TOP-5 INTERVALS
# ADAPTED FOR YOUR JSON STRUCTURE
# ======================================================

import os
import json
import networkx as nx
from classes.LLMUser import LLMUser
from classes.AF import *
from classes.ServerOllama import OllamaServer, OllamaChat
from classes.PromptBuilder import *
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms

# ======================================================
# ARGUMENTATION GRAPH CLASS
# ======================================================

class ArgumentationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()
        self.node_text_map = {}

    # Add argument nodes
    def add_argument(self, arg_id: str, text: str, node_type: str = "evidence", initial_strength: float = 0.5):
        self.G.add_node(arg_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[arg_id] = Argument(arg_id, initial_weight=initial_strength)
        self.node_text_map[arg_id] = text

    # Add relations (attack or support)
    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

    # Compute argument strengths
    def compute_strengths(self, delta: float = 1e-2, epsilon: float = 1e-4):
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)

        # Normalize BAG arguments
        self.bag.arguments = {arg_obj: arg_obj for arg_obj in self.bag.arguments.values()}

        # Reconstruct attacker/supporter lists
        new_attacker = {}
        new_supporter = {}
        for arg in self.bag.arguments.keys():
            if hasattr(self.bag, "attacker") and arg in self.bag.attacker:
                new_attacker[arg] = [
                    a if isinstance(a, Argument) else self.bag.arguments.get(a)
                    for a in self.bag.attacker[arg]
                    if a is not None
                ]
            else:
                new_attacker[arg] = []

            if hasattr(self.bag, "supporter") and arg in self.bag.supporter:
                new_supporter[arg] = [
                    s if isinstance(s, Argument) else self.bag.arguments.get(s)
                    for s in self.bag.supporter[arg]
                    if s is not None
                ]
            else:
                new_supporter[arg] = []

        self.bag.attacker = new_attacker
        self.bag.supporter = new_supporter

        # Solve model
        arg_model.solve(delta=delta, epsilon=epsilon, verbose=False)

        # Extract strengths
        strengths = {}
        for arg in self.bag.arguments:
            if hasattr(arg_model, "acceptability") and arg in arg_model.acceptability:
                strengths[str(arg.name)] = arg_model.acceptability[arg]
            elif hasattr(arg, "strength"):
                strengths[str(arg.name)] = arg.strength
            else:
                strengths[str(arg.name)] = arg.initial_weight

        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # Build argumentation graph from text
    def build_from_text(self, text: str, llm_user: LLMUser, extra_arguments: list = None, insert_at_start: bool = False, max_arguments: int = 5):
        # Extract arguments from text
        arguments = llm_user.extract_arguments_with_ollama(text)

        # Merge extra arguments
        if extra_arguments:
            arguments = extra_arguments + arguments if insert_at_start else arguments + extra_arguments

        # Limit number of arguments
        if max_arguments is not None:
            arguments = arguments[:max_arguments]

        # Detect pairwise relations
        relations_dict = llm_user.detect_argument_relations_pairwise(arguments)

        # Add nodes
        for i, arg_text in enumerate(arguments):
            node_type = "claim" if i == 0 else "evidence"
            self.add_argument(str(i), arg_text, node_type=node_type)

        # Add relations
        for key, rel in relations_dict.items():
            i, j = key.split("-")
            if rel in ["support", "attack"]:
                self.add_relation(i, j, rel)

        # Compute strengths
        strengths = self.compute_strengths()
        return {"graph": self.G, "strengths": strengths, "node_text_map": self.node_text_map}

    # Utility getters
    def get_text_from_id(self, node_id):
        return self.node_text_map.get(node_id, None)

    def get_id_from_text(self, text):
        for nid, t in self.node_text_map.items():
            if t == text:
                return nid
        return None

# ======================================================
# CONFIGURATION
# ======================================================

MODEL_NAME = "gpt-oss:20b"  # Ollama model
DATASET_FILE = "dataset/your_file.json"  # Replace with your JSON
OUTPUT_DIR = os.path.join("results", MODEL_NAME.replace(":", "_"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize server, LLM, and graph
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

for i, entry in enumerate(dataset):
    print(f"\n=== Processing Question {i+1} ===")
    question = entry["question"]
    possible_answers = entry["choices"]
    correct_answer = entry["correct_answer"]

    # Hypotheses
    dict_hypthesis = {choice: choice for choice in possible_answers}
    hypotheses = list(dict_hypthesis.values())

    # Extract top intervals from ranked_pages
    intervals = []
    for page, page_data in entry.get("ranked_pages", {}).items():
        for interval_entry in page_data.get("summary_ranked", []):
            interval_text = interval_entry.get("text", "").strip()
            score = interval_entry.get("score", 0)
            if interval_text:
                intervals.append({"interval": interval_text, "score": score})

    # Sort intervals by descending score
    intervals = sorted(intervals, key=lambda x: x["score"], reverse=True)
    combined_text = ""
    predicted_answer = None
    strengths_text = {}

    # Progressive integration: top 5 intervals
    for step_start in range(0, len(intervals), 5):
        top_intervals = intervals[step_start: step_start + 5]
        step_text = " ".join(interval["interval"] for interval in top_intervals)
        combined_text += step_text + " "

        graph_result = graph_builder.build_from_text(
            text=combined_text,
            llm_user=llm_user,
            extra_arguments=hypotheses,
            insert_at_start=True,
            max_arguments=5
        )
        strengths = graph_result.get("strengths", {})
        strengths_text = {}

        for nid, strength_val in strengths.items():
            node_text = graph_builder.get_text_from_id(nid)
            for hyp in hypotheses:
                if hyp.lower() in node_text.lower():
                    strengths_text[hyp] = strength_val

        if not strengths_text:
            continue

        sorted_strengths = sorted(strengths_text.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_strengths) > 1 and (sorted_strengths[0][1] - sorted_strengths[1][1]) > 0.15:
            predicted_answer = sorted_strengths[0][0]
            break
        elif len(sorted_strengths) == 1:
            predicted_answer = sorted_strengths[0][0]
            break

    # Fallback
    if not predicted_answer and strengths_text:
        predicted_answer = max(strengths_text, key=strengths_text.get)

    y_true.append(correct_answer)
    y_pred.append(predicted_answer)
    combined_data.append({
        "question": question,
        "hypotheses": hypotheses,
        "predicted_answer": predicted_answer,
        "correct_answer": correct_answer,
        "strengths": strengths_text,
        "used_intervals": [interval["interval"] for interval in intervals[:5]]
    })

    # Save graph
    graph_file = os.path.join(OUTPUT_DIR, f"graph_question_{i+1}.json")
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(graph_result["graph"]), f, indent=2)
    print(f"Graph saved to {graph_file}")
    print(f"✅ Correct: {correct_answer} 🔮 Predicted: {predicted_answer}")
    print("Hypotheses strengths:", strengths_text)

# ======================================================
# SAVE RESULTS
# ======================================================

with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "y_true_y_pred.json"), "w", encoding="utf-8") as f:
    json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=2, ensure_ascii=False)

# Compute accuracy
correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
accuracy = correct_count / len(y_true) if y_true else 0.0

print("\n========================================")
print(f"🏁 Finished! Total evaluated: {len(y_true)}")
print(f"✅ Correct: {correct_count}")
print(f"📊 Accuracy: {accuracy*100:.2f}%")
print(f"Results saved in: {OUTPUT_DIR}")
print("========================================")
