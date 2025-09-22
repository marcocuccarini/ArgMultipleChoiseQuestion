import networkx as nx
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms
from classes.LLMUser import LLMUser  # adjust import path if needed


class ArgumentationGraph:
    """
    Argumentation graph class:
    - Adds arguments and relations
    - Computes argument strengths
    - Builds graph from text using LLMUser
    - Allows adding extra arguments at start or end
    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.bag = BAG()
        self.node_text_map = {}  # node_id -> text

    # --- Add nodes ---
    def add_argument(
        self,
        arg_id: str,
        text: str,
        node_type: str = "evidence",
        initial_strength: float = 0.5,
    ):
        self.G.add_node(arg_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[arg_id] = Argument(arg_id, initial_weight=initial_strength)
        self.node_text_map[arg_id] = text

    # --- Add edges ---
    def add_relation(self, src: str, tgt: str, relation: str):
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])

    # --- Compute strengths ---
    def compute_strengths(self, delta: float = 1e-2, epsilon: float = 1e-4):
        """
        Compute argument strengths using ContinuousDFQuADModel with RK4.
        After solve(), strengths are stored in BAG arguments (initial_weight).
        """
        arg_model = semantics.ContinuousDFQuADModel()
        arg_model.BAG = self.bag
        arg_model.approximator = algorithms.RK4(arg_model)
        arg_model.solve(delta=delta, epsilon=epsilon)

        # Strengths from BAG arguments
        strengths = {
            arg_id: self.bag.arguments[arg_id].initial_weight
            for arg_id in self.bag.arguments
        }

        # Update networkx node attributes
        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # --- Build graph from text ---
    def build_from_text(
        self,
        text: str,
        llm_user: LLMUser,
        extra_arguments: list = None,
        insert_at_start: bool = False
    ) -> dict:
        """
        Build an argumentation graph from text, optionally adding extra arguments.
        :param text: main text
        :param llm_user: LLMUser instance
        :param extra_arguments: list of hypothesis strings
        :param insert_at_start: if True, extra_arguments are inserted before extracted arguments
        """
        # Extract arguments from text
        arguments = llm_user.extract_arguments_with_ollama(text)

        # Add extra hypotheses if provided
        if extra_arguments:
            if insert_at_start:
                arguments = extra_arguments + arguments
            else:
                arguments = arguments + extra_arguments

        # Detect pairwise argument relations
        relations_dict = llm_user.detect_argument_relations_pairwise(arguments)

        # Add nodes
        for i, arg_text in enumerate(arguments):
            node_type = "claim" if i == 0 else "evidence"
            self.add_argument(str(i), arg_text, node_type=node_type)

        # Add edges based on detected relations
        for key, rel in relations_dict.items():
            i, j = key.split("-")
            if rel in ["support", "attack"]:
                self.add_relation(i, j, rel)

        # Compute argument strengths
        strengths = self.compute_strengths()

        return {
            "graph": self.G,
            "strengths": strengths,
            "node_text_map": self.node_text_map
        }

    # --- Helper methods ---
    def get_text_from_id(self, node_id):
        return self.node_text_map.get(node_id, None)

    def get_id_from_text(self, text):
        for nid, t in self.node_text_map.items():
            if t == text:
                return nid
        return None
