 # argument_graph.py

import networkx as nx
import re
import json
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms
from classes.LLMUser import LLMUser


def clean_json_response(raw_response: str) -> str:
    """
    Rimuove blocchi di codice Markdown (```json ... ```) e spazi extra.
    """
    cleaned = re.sub(r"```(?:json)?\n(.*?)```", r"\1", raw_response, flags=re.DOTALL)
    return cleaned.strip()


class ArgumentationGraph:
    """
    Classe per costruire grafi argomentativi:
    - Estrae argomenti e relazioni con LLMUser
    - Calcola la forza degli argomenti con DF-QuAD
    - Esporta un grafo in JSON
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()

    # Aggiungi nodo
    def add_argument(self, arg_id: str, text: str, node_type: str = "evidence", initial_strength: float = 0.5):
        self.G.add_node(arg_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[arg_id] = Argument(arg_id, initial_weight=initial_strength)

    # Aggiungi arco
    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

    # Calcola forze con DF-QuAD
    def compute_strengths(self, delta: float = 1e-2, epsilon: float = 1e-4):
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)
        arg_model.solve(delta=delta, epsilon=epsilon)

        strengths = {a.name: a.strength for a in self.bag.arguments.values()}
        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # Esporta grafo
    def get_graph(self):
        return self.G

    def to_json(self):
        return {
            "nodes": [
                {"id": n, "type": self.G.nodes[n]["type"], "text": self.G.nodes[n]["text"], "strength": self.G.nodes[n]["strength"]}
                for n in self.G.nodes
            ],
            "edges": [
                {"source": u, "target": v, "relation": d["relation"]}
                for u, v, d in self.G.edges(data=True)
            ]
        }

    # Pipeline completa
    def build_from_text(self, text: str, llm_user: LLMUser) -> dict:
        # 🔹 1. Estrai argomenti
        raw_args = llm_user.extract_arguments_with_ollama(text)
        arguments = []
        for arg in raw_args:
            if isinstance(arg, str):
                cleaned = clean_json_response(arg)
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        arguments.extend(parsed)
                    else:
                        arguments.append(parsed)
                except Exception:
                    arguments.append(cleaned)
            else:
                arguments.append(arg)

        # 🔹 2. Estrai relazioni
        raw_relations = llm_user.detect_argument_relations_pairwise(arguments)
        relations_dict = {}
        for key, val in raw_relations.items():
            if not isinstance(key, str) or "-" not in key:
                print(f"[WARN] Skipping invalid relation key: {key}")
                continue

            cleaned_val = clean_json_response(val) if isinstance(val, str) else val
            if cleaned_val in ["support", "attack"]:
                relations_dict[key] = cleaned_val
            else:
                print(f"[WARN] Skipping unknown relation type: {cleaned_val}")

        # 🔹 3. Aggiungi nodi
        for i, arg_text in enumerate(arguments):
            node_type = "claim" if i == 0 else "evidence"
            self.add_argument(str(i), arg_text, node_type=node_type)

        # 🔹 4. Aggiungi archi in modo sicuro
        for key, rel in relations_dict.items():
            try:
                i, j = key.split("-", 1)
                self.add_relation(i, j, rel)
            except ValueError:
                print(f"[WARN] Could not parse relation key: {key}")
                continue

        # 🔹 5. Calcola forze
        strengths = self.compute_strengths()

        return {
            "arguments": arguments,
            "relations": relations_dict,
            "graph": self.to_json(),
            "strengths": strengths
        }
