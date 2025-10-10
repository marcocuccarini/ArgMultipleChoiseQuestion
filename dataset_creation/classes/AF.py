import networkx as nx
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms
from classes.LLMUser import LLMUser  # adjust import path if needed

# ======================================================
# ARGUMENTATION GRAPH CLASS
# ======================================================

class ArgumentationGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()
        self.node_text_map = {}

    # --------------------------------------------------
    # Add argument nodes
    # --------------------------------------------------
    def add_argument(self, arg_id: str, text: str, node_type: str = "evidence", initial_strength: float = 0.5):
        self.G.add_node(arg_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[arg_id] = Argument(arg_id, initial_weight=initial_strength)
        self.node_text_map[arg_id] = text

    # --------------------------------------------------
    # Add relations (attack or support)
    # --------------------------------------------------
    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

    # --------------------------------------------------
    # Compute argument strengths (FIXED VERSION)
    # --------------------------------------------------
    def compute_strengths(self, delta: float = 1e-2, epsilon: float = 1e-4):
        # --- build model ---
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)

        # --- normalize BAG argument keys ---
        self.bag.arguments = {arg_obj: arg_obj for arg_obj in self.bag.arguments.values()}

        # reconstruct attacker/supporter lists
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

        # --- solve the model ---
        arg_model.solve(delta=delta, epsilon=epsilon)

        # --- extract FINAL strengths ---
        strengths = {}
        for arg in self.bag.arguments:
            if hasattr(arg_model, "acceptability") and arg in arg_model.acceptability:
                strengths[str(arg.name)] = arg_model.acceptability[arg]
            elif hasattr(arg, "strength"):
                strengths[str(arg.name)] = arg.strength
            else:
                strengths[str(arg.name)] = arg.initial_weight  # fallback

        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # --------------------------------------------------
    # Build argumentation graph directly from text
    # --------------------------------------------------
    def build_from_text(self, text: str, llm_user: LLMUser, extra_arguments: list = None, insert_at_start: bool = False):
        arguments = llm_user.extract_arguments_with_ollama(text)
        if extra_arguments:
            arguments = extra_arguments + arguments if insert_at_start else arguments + extra_arguments

        relations_dict = llm_user.detect_argument_relations_pairwise(arguments)

        for i, arg_text in enumerate(arguments):
            node_type = "claim" if i == 0 else "evidence"
            self.add_argument(str(i), arg_text, node_type=node_type)

        for key, rel in relations_dict.items():
            i, j = key.split("-")
            if rel in ["support", "attack"]:
                self.add_relation(i, j, rel)

        strengths = self.compute_strengths()
        return {"graph": self.G, "strengths": strengths, "node_text_map": self.node_text_map}

    # --------------------------------------------------
    # Utility getters
    # --------------------------------------------------
    def get_text_from_id(self, node_id):
        return self.node_text_map.get(node_id, None)

    def get_id_from_text(self, text):
        for nid, t in self.node_text_map.items():
            if t == text:
                return nid
        return None