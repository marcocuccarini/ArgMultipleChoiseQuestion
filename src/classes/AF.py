import networkx as nx
from classes.LLMUser import LLMUser
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms

class ArgumentationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()
        self.node_text_map = {}

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

    # Compute strengths
    def compute_strengths(self, delta=1e-2, epsilon=1e-4):
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)
        arg_model.solve(delta=delta, epsilon=epsilon, verbose=True)
        strengths = {str(arg.name): getattr(arg_model, 'acceptability', {}).get(arg, arg.initial_weight) for arg in self.bag.arguments.values()}
        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # Build graph with hypotheses + text arguments
    def build_from_text(self, text: str, llm_user: LLMUser, hypotheses: list = None, max_arguments: int = 5):
        if hypotheses is None:
            hypotheses = []

        # Step 1: add hypotheses
        for i, hyp_text in enumerate(hypotheses):
            self.add_argument(f"H{i}", hyp_text, node_type="hypothesis", initial_strength=0.5)

        # Step 2: extract arguments from text
        arguments = llm_user.extract_arguments_with_ollama(text)
        if max_arguments:
            arguments = arguments[:max_arguments]

        # Step 3: add arguments
        node_offset = len(self.G.nodes)
        for i, arg_text in enumerate(arguments):
            arg_id = f"A{i+node_offset}"
            self.add_argument(arg_id, arg_text, node_type="argument")

        # Step 4: detect relations
        all_texts = list(self.node_text_map.values())
        relations_dict = llm_user.detect_argument_relations_pairwise(all_texts)
        for key, rel in relations_dict.items():
            i, j = key.split("-")
            if rel in ["support", "attack"]:
                # Hypotheses cannot attack each other
                if self.G.nodes[i]["type"] == "hypothesis" and self.G.nodes[j]["type"] == "hypothesis":
                    continue
                self.add_relation(i, j, rel)

        # Step 5: compute strengths
        strengths = self.compute_strengths()
        return {"graph": self.G, "strengths": strengths, "node_text_map": self.node_text_map}

    # Extend graph with new text arguments
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
        relations_dict = llm_user.detect_argument_relations_pairwise(all_texts)
        for key, rel in relations_dict.items():
            i, j = key.split("-")
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
