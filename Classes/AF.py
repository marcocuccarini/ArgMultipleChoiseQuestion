import networkx as nx

from classes.LLMUser import LLMUser  # adjust imports if needed

import networkx as nx

class ArgumentationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.argument_map = {}   # text → node_id
        self.id_map = {}         # node_id → text
        self.next_id = 0         # auto-increment node IDs

    def _add_argument(self, text, arg_type="argument"):
        node_id = self.next_id
        self.graph.add_node(node_id, text=text, type=arg_type)
        self.argument_map[text] = node_id
        self.id_map[node_id] = text
        self.next_id += 1
        return node_id

    def build_from_text(self, text, user=None, extra_arguments=None, insert_at_start=False):
        """
        Build a graph from main text + optional hypotheses.
        Returns {'graph': DiGraph, 'strengths': {...}}
        """
        self.graph.clear()
        self.argument_map.clear()
        self.id_map.clear()
        self.next_id = 0

        # Add hypotheses first
        if extra_arguments:
            for hyp in extra_arguments:
                if hyp and isinstance(hyp, str):
                    self._add_argument(hyp, arg_type="hypothesis")

        # Parse main text into arguments
        parsed_args = self.parse_text_into_arguments(text, user)
        for arg in parsed_args:
            if arg and isinstance(arg, str):
                self._add_argument(arg, arg_type="argument")

        # Build edges (dummy logic; replace with real attack/support detection)
        self._build_relations()

        # Compute strengths via Uncertainpy solver
        strengths = self.compute_strengths()

        return {"graph": self.graph, "strengths": strengths}

    def parse_text_into_arguments(self, text, user=None):
        """Dummy parser: splits text into sentences."""
        if not text or not text.strip():
            return []
        return [sent.strip() for sent in text.split(".") if sent.strip()]

    def _build_relations(self):
        """Dummy edges: chain all nodes as attacks."""
        nodes = list(self.graph.nodes)
        for i in range(len(nodes) - 1):
            self.graph.add_edge(nodes[i], nodes[i+1], type="attack")

    def compute_strengths(self, delta=0.01, epsilon=1e-6):
        """
        Convert NetworkX graph to ADS/BAG for Uncertainpy and compute strengths.
        """
        # Import Uncertainpy classes
        from core_engine.Uncertainpy.src.uncertainpy.gradual.semantics.ADS import ADS
        from core_engine.Uncertainpy.src.uncertainpy.gradual.semantics.Model import Model
        from core_engine.Uncertainpy.src.uncertainpy.gradual.algorithms.RK4 import RK4

        # --- Step 1: Create ADS and add arguments ---
        ads = ADS()
        for node_id in self.graph.nodes:
            text = self.id_map.get(node_id, f"arg_{node_id}")
            ads.add_argument(node_id, text=text)

        # --- Step 2: Add attacks/supports ---
        for source, target, data in self.graph.edges(data=True):
            edge_type = data.get("type", "attack")
            if edge_type == "attack":
                ads.add_attack(source, target)
            elif edge_type == "support":
                ads.add_support(source, target)

        # --- Step 3: Create model + approximator ---
        arg_model = Model(ads)
        approximator = RK4(arg_model)
        arg_model.approximator = approximator

        # --- Step 4: Solve ---
        arg_model.solve(delta=delta, epsilon=epsilon)

        # --- Step 5: Return strengths keyed by node_id ---
        return arg_model.get_strengths()

    # --- Helper methods to map IDs ↔ texts ---
    def get_text_from_id(self, node_id):
        return self.id_map.get(node_id, None)

    def get_id_from_text(self, text):
        return self.argument_map.get(text, None)
