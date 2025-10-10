# llm_user.py
import json
from classes.classes_server_ollama import *
from classes.prompt import ARGUMENT_EXTRACTION_PROMPT, PAIRWISE_RELATION_PROMPT
from classes.utils import clean_json_response

# classes/LLM.py
from classes.classes_server_ollama import OllamaServer, OllamaChat, LLMResponse

class LLM:
    def __init__(self, model: str):
        self.server = OllamaServer()
        self.model = model
        self.chat = OllamaChat(self.server, model)

    def run_inference(self, prompt: str) -> str:
        """Send prompt to the model and return raw text response."""
        response = self.chat.send_prompt(prompt)
        if response.response_type == "generated":
            return response.raw_text
        else:
            raise RuntimeError(f"LLM Error: {response.raw_text}")


class LLMUser:
    """
    High-level interface for:
    - verbalizing choices into facts
    - extracting arguments from text
    - detecting pairwise argument relations
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    # ---------------------------
    # Fact verbalization
    # ---------------------------
    def verbalize_choice(self, question: str, choice: str) -> str:
        prompt = f"""
        Transform the following multiple-choice question and option into a concise factual statement,
        making the option the subject of the sentence.

        Example:
        Question: "Compounds that are capable of accepting electrons, such as O2 or F2, are called what?"
        Option: "oxidants"
        Output: "Oxidants are compounds capable of accepting electrons, such as O2 or F2."

        Now transform this one:

        Question: "{question}"
        Option: "{choice}"

        Return only the factual statement as plain text.
        """
        fact = self.llm.run_inference(prompt)
        # --- strip formatting if model returns json or code blocks ---
        return fact



    # ---------------------------
    # Argument extraction
    # ---------------------------
    def extract_arguments_with_ollama(self, text: str) -> list:
        """
        Extract arguments from text using the LLM and prompt.
        Returns a list of strings or dicts.
        """
        prompt = ARGUMENT_EXTRACTION_PROMPT.format(text=text)
        raw_response = self.llm.run_inference(prompt)

        try:
            arguments = json.loads(raw_response)
            if isinstance(arguments, list):
                return arguments
            else:
                return [str(arguments)]
        except Exception as e:
            print(f"[WARN] Failed to parse JSON from argument extraction: {e}")
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

                prompt = PAIRWISE_RELATION_PROMPT.format(arg_a=arg_a, arg_b=arg_b)
                raw_response = self.llm.run_inference(prompt)

                try:
                    result = json.loads(clean_json_response(raw_response))
                    rel = result.get("relation", "indifferent")
                    relations[f"{i}-{j}"] = rel
                except Exception as e:
                    print(f"[WARN] Failed relation parse for {i}-{j}: {e}")
                    relations[f"{i}-{j}"] = "indifferent"

        return relations
