# ======================================================
# ARGUMENTATION GRAPH CLASS
# ======================================================
import os
import json
import networkx as nx
from classes.LLMUser import LLMUser
from classes.AF import *
from classes.ServerOllama import OllamaServer, OllamaChat
from classes.PromptBuilder import *
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms

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


    def extend_from_text(self, text: str, llm_user: LLMUser, max_arguments: int = 5):

        """
        Extend the existing argumentation graph with new arguments extracted from text.
        Does NOT reset the current graph.
        """
        # Step 1. Extract new arguments from the text
        new_arguments = llm_user.extract_arguments_with_ollama(text)
        if not new_arguments:
            print("⚠️ No new arguments extracted from text.")
            return {"graph": self.G, "strengths": {}, "node_text_map": self.node_text_map}

        # Step 2. Limit number of arguments if requested
        if max_arguments:
            new_arguments = new_arguments[:max_arguments]

        # Step 3. Combine all argument texts (old + new)
        existing_texts = list(self.node_text_map.values())
        all_texts = existing_texts + new_arguments

        # Step 4. Detect relations between all arguments
        relations_dict = llm_user.detect_argument_relations_pairwise(all_texts)

        # Step 5. Add new argument nodes
        node_offset = len(self.G.nodes)
        for i, arg_text in enumerate(new_arguments):
            arg_id = str(node_offset + i)
            self.add_argument(arg_id, arg_text, node_type="evidence")

        # Step 6. Add new support/attack edges
        for key, rel in relations_dict.items():
            i, j = key.split("-")
            if rel in ["support", "attack"]:
                # Only add relation if both nodes exist
                if i in self.G.nodes and j in self.G.nodes:
                    self.add_relation(i, j, rel)

        # Step 7. Recompute argument strengths
        strengths = self.compute_strengths()

        print(f"✅ Extended graph with {len(new_arguments)} new arguments.")
        return {"graph": self.G, "strengths": strengths, "node_text_map": self.node_text_map}

    # Utility getters
    def get_text_from_id(self, node_id):
        return self.node_text_map.get(node_id, None)

    def get_id_from_text(self, text):
        for nid, t in self.node_text_map.items():
            if t == text:
                return nid
        return None