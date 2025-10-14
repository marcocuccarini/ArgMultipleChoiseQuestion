import os
import json
import networkx as nx
from classes.LLMUser import LLMUser
from classes.ServerOllama import OllamaServer, OllamaChat
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms

# ======================================================
# ARGUMENTATION GRAPH CLASS
# ======================================================
class ArgumentationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()
        self.node_text_map = {}  # node_id -> text

    # Add a node
    def add_argument(self, arg_id: str, text: str, node_type: str = "argument", initial_strength: float = 0.5):
        self.G.add_node(arg_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[arg_id] = Argument(arg_id, initial_weight=initial_strength)
        self.node_text_map[arg_id] = text

    # Add support/attack edge
    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

    # Compute strengths using BAG
    def compute_strengths(self, delta=1e-2, epsilon=1e-4):
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)
        arg_model.solve(delta=delta, epsilon=epsilon, verbose=True)
        strengths = {str(arg.name): getattr(arg_model, 'acceptability', {}).get(arg, arg.initial_weight) for arg in self.bag.arguments.values()}
        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # Build graph from hypotheses + text
    def build_from_text(self, text: str, llm_user: LLMUser, hypotheses: list = None, max_arguments: int = 5):
        if hypotheses is None:
            hypotheses = []

        # Step 1: Add hypotheses
        for i, hyp_text in enumerate(hypotheses):
            self.add_argument(f"H{i}", hyp_text, node_type="hypothesis", initial_strength=0.5)

        # Step 2: Extract arguments from text
        arguments = llm_user.extract_arguments_with_ollama(text)
        if max_arguments:
            arguments = arguments[:max_arguments]

        # Step 3: Add arguments
        node_offset = len(self.G.nodes)
        for i, arg_text in enumerate(arguments):
            arg_id = f"A{i+node_offset}"
            self.add_argument(arg_id, arg_text, node_type="argument")

        # Step 4: Detect relations
        all_texts = list(self.node_text_map.values())
        all_node_ids = list(self.node_text_map.keys())
        relations_dict = llm_user.detect_argument_relations_pairwise(all_texts)
        for key, rel in relations_dict.items():
            idx_i, idx_j = key.split("-")
            i = all_node_ids[int(idx_i)]
            j = all_node_ids[int(idx_j)]
            if rel in ["support", "attack"]:
                # Hypotheses cannot attack each other
                if self.G.nodes[i]["type"] == "hypothesis" and self.G.nodes[j]["type"] == "hypothesis":
                    continue
                self.add_relation(i, j, rel)

        # Step 5: Compute strengths
        strengths = self.compute_strengths()
        return {"graph": self.G, "strengths": strengths, "node_text_map": self.node_text_map}

    # Extend graph with new text
    def extend_from_text(self, text: str, llm_user: LLMUser, max_arguments: int = 5):
        new_arguments = llm_user.extract_arguments_with_ollama(text)
        if not new_arguments:
            print("⚠️ No new arguments extracted from text.")
            return {"graph": self.G, "strengths": {}, "node_text_map": self.node_text_map}

        if max_arguments:
            new_arguments = new_arguments[:max_arguments]

        node_offset = len(self.G.nodes)
        for i, arg_text in enumerate(new_arguments):
            arg_id = f"A{i+node_offset}"
            self.add_argument(arg_id, arg_text, node_type="argument")

        # Detect relations among all nodes
        all_texts = list(self.node_text_map.values())
        all_node_ids = list(self.node_text_map.keys())
        relations_dict = llm_user.detect_argument_relations_pairwise(all_texts)
        for key, rel in relations_dict.items():
            idx_i, idx_j = key.split("-")
            i = all_node_ids[int(idx_i)]
            j = all_node_ids[int(idx_j)]
            if rel in ["support", "attack"]:
                if self.G.nodes[i]["type"] == "hypothesis" and self.G.nodes[j]["type"] == "hypothesis":
                    continue
                self.add_relation(i, j, rel)

        strengths = self.compute_strengths()
        return {"graph": self.G, "strengths": strengths, "node_text_map": self.node_text_map}

    # Utility
    def get_text_from_id(self, node_id):
        return self.node_text_map.get(node_id, None)

    def get_id_from_text(self, text):
        for nid, t in self.node_text_map.items():
            if t == text:
                return nid
        return None

# ======================================================
# MAIN PIPELINE
# ======================================================
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "gpt-oss:20b"
    RETRIEVE_VALUE = 5
    CONFIDENCE_THRESHOLD = 0.15
    MAX_HYPOTHESES = 20
    DATASET_FILE = "dataset/wiki_ranked_pages.json"
    OUTPUT_DIR = os.path.join("results", MODEL_NAME.replace(":", "_"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize
    server = OllamaServer()
    llm = OllamaChat(server, MODEL_NAME)
    llm_user = LLMUser(llm)
    graph_builder = ArgumentationGraph()

    # Load dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        if isinstance(dataset, dict):
            dataset = [dataset]

    y_true = []
    y_pred = []
    combined_data = []

    for idx, entry in enumerate(dataset):
        print(f"\n=== Processing Question {idx + 1} ===")
        question = entry.get("question", "")
        correct_answer = entry.get("correct_answer", "")
        hypotheses = entry.get("choice_facts", "").values()
        
        predicted_answer = None
        strengths_text = {}
        graph_result = None

        ranked_pages = entry.get("ranked_pages", {})
        if not ranked_pages:
            print(f"⚠️ No ranked pages for question {idx + 1}")
            continue

      

        # Step 2: Iteratively build/extend argumentation graph
        for page_name, page_data in ranked_pages.items():
            for section in ["summary_ranked", "other_ranked"]:
                paragraphs = page_data.get(section, [])
                paragraphs = sorted(paragraphs, key=lambda x: x.get("score", 0), reverse=True)
                top_texts = [p.get("text", "").strip() for p in paragraphs[:RETRIEVE_VALUE] if p.get("text")]
                combined_text = " ".join(top_texts).strip()
                if not combined_text:
                    continue
                print(f"📄 Using {len(combined_text.split())} words from page '{page_name}' [{section}]")

                
                try:
                    if graph_result is None:

                        print("Combined text", combined_text)
                        print("hypothesis", hypotheses)

                        
                        graph_result = graph_builder.build_from_text(
                            text=combined_text,
                            llm_user=llm_user,
                            hypotheses=hypotheses
                        )
                    else:
                        graph_result = graph_builder.extend_from_text(
                            text=combined_text,
                            llm_user=llm_user
                        )
                except Exception as e:
                    print(f"❌ Error building/extending graph: {e}")
                    continue

                if not graph_result or "graph" not in graph_result:
                    continue

                # Step 3: Compute hypothesis strengths
                strengths = graph_result.get("strengths", {})
                for nid, strength_val in strengths.items():
                    node_text = graph_builder.get_text_from_id(nid)
                    if not node_text:
                        continue
                    for hyp in hypotheses:
                        if hyp.lower() in node_text.lower():
                            strengths_text[hyp] = strength_val

                # Early exit if confident
                if strengths_text:
                    sorted_strengths = sorted(strengths_text.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_strengths) == 1 or (sorted_strengths[0][1] - sorted_strengths[1][1]) > CONFIDENCE_THRESHOLD:
                        predicted_answer = sorted_strengths[0][0]
                        break
            if predicted_answer:
                break

        # Step 4: Fallback
        if not predicted_answer:
            predicted_answer = max(strengths_text, key=strengths_text.get) if strengths_text else "Unknown"

        # Step 5: Save results
        y_true.append(correct_answer)
        y_pred.append(predicted_answer)
        combined_data.append({
            "question": question,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "strengths": strengths_text
        })

        if graph_result and "graph" in graph_result:
            graph_file = os.path.join(OUTPUT_DIR, f"graph_question_{idx + 1}.json")
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(nx.node_link_data(graph_result["graph"]), f, indent=2)
            print(f"📁 Graph saved to {graph_file}")

        print(f"✅ Correct: {correct_answer} 🔮 Predicted: {predicted_answer}")
        print("Hypotheses strengths:", strengths_text)

    # ======================================================
    # Save results and accuracy
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
