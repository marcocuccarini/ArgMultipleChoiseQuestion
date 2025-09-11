# argument_graph.py

import networkx as nx
from src.uncertainpy.gradual import Argument, BAG, semantics, algorithms

class ArgumentationGraph:
    """
    Argumentation graph class that:
    - Adds arguments and relations
    - Computes argument strengths
    - Builds a graph from raw text using LLMUser
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()

    # Add nodes
    def add_argument(self, arg_id: str, text: str, node_type: str = "evidence", initial_strength: float = 0.5):
        self.G.add_node(arg_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[arg_id] = Argument(arg_id, initial_weight=initial_strength)

    # Add edges
    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

    # Compute semantics
    def compute_strengths(self, delta: float = 1e-2, epsilon: float = 1e-4):
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)
        arg_model.solve(delta=delta, epsilon=epsilon)

        strengths = {a.name: a.strength for a in self.bag.arguments.values()}
        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # Export graph
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

    # Full pipeline: build graph from text
    def build_from_text(self, text: str, llm_user: LLMUser) -> dict:
        # Extract arguments
        arguments = llm_user.extract_arguments_with_ollama(text)

        # Detect relations
        relations_dict = llm_user.detect_argument_relations(arguments)

        # Add nodes
        for i, arg_text in enumerate(arguments):
            node_type = "claim" if i == 0 else "evidence"  # first argument as claim
            self.add_argument(str(i), arg_text, node_type=node_type)

        # Add edges
        for key, rel in relations_dict.items():
            i, j = key.split("-")
            if rel in ["support", "attack"]:
                self.add_relation(i, j, rel)

        # Compute strengths
        strengths = self.compute_strengths()

        return {
            "arguments": arguments,
            "relations": relations_dict,
            "graph": self.to_json(),
            "strengths": strengths
        }
