import json
import re
from classes.PromptBuilder import PromptBuilder

class LLMUser:
    """
    High-level interface for:
    - verbalizing choices into facts
    - extracting arguments from text
    - detecting pairwise argument relations
    """

    def __init__(self, llm):
        self.llm = llm

    # ---------------------------
    # Wikipedia retrieval
    # ---------------------------
    def get_candidate_pages(self, question, choices, max_pages=5):
        """
        Generate a list of relevant Wikipedia page titles using the LLM.
        """
        prompt = PromptBuilder.wikipedia_retrieval_prompt(question, choices, max_pages)
        response = self.llm.send_prompt(prompt)

        try:
            pages = json.loads(response.raw_text)
            if isinstance(pages, list):
                return pages
        except Exception:
            pass
        
        return [response.raw_text]

    # ---------------------------
    # Argument extraction
    # ---------------------------
    def extract_arguments_with_ollama(self, text: str) -> list:
        """
        Extract arguments from text using the LLM.
        Returns a list of argument strings.
        """
        prompt = PromptBuilder.argument_extraction_prompt(text)
        raw_response = self.llm.send_prompt(prompt).raw_text.strip()

        # Try JSON parsing first
        try:
            arguments = json.loads(raw_response)
            if isinstance(arguments, list):
                return [a.strip() for a in arguments if isinstance(a, str) and len(a.strip()) > 2]
            elif isinstance(arguments, dict):
                return [v.strip() for v in arguments.values() if isinstance(v, str) and len(v.strip()) > 2]
        except Exception:
            pass

        # Fallbacks
        lines = [l.strip("-• \t") for l in raw_response.split("\n") if len(l.strip()) > 3]
        if len(lines) > 1:
            return lines

        sentences = re.split(r'(?<=[.!?])\s+', raw_response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 4]
        if sentences:
            return sentences

        return [raw_response]

    # ---------------------------
    # Relation detection
    # ---------------------------
    def detect_argument_relations_pairwise(self, arguments: list) -> dict:
        """
        Compare all pairs of arguments and detect relations.
        Returns a dict like {"0-1": "support", "1-2": "attack", ...}.
        """
        relations = {}
        for i, arg_a in enumerate(arguments):
            for j, arg_b in enumerate(arguments):
                if i == j:
                    continue

                prompt = PromptBuilder.pairwise_relation_prompt(arg_a, arg_b)
                raw_response = self.llm.send_prompt(prompt).raw_text

                try:
                    result = json.loads(raw_response)
                    rel = result.get("relation", "indifferent")
                    relations[f"{i}-{j}"] = rel
                except Exception:
                    relations[f"{i}-{j}"] = "indifferent"

        return relations
