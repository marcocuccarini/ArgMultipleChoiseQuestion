class LLMUser:
    """
    High-level class that performs tasks using an LLM instance.
    """

    def __init__(self, llm: LLM):
        self.llm = llm

    def extract_arguments_with_ollama(self, text: str) -> list:
        """
        Extract arguments (claims/premises) from raw text using LLM.
        """
        from prompts import ARGUMENT_EXTRACTION_PROMPT
        prompt = ARGUMENT_EXTRACTION_PROMPT.format(text=text)
        raw_response = self.llm.run_inference(prompt)

        try:
            arguments = json.loads(raw_response)
            if isinstance(arguments, list):
                return arguments
            else:
                return [str(arguments)]
        except Exception as e:
            print(f"[WARN] Failed to parse JSON: {e}")
            return [raw_response]

    def detect_argument_relations(self, arguments: list) -> dict:
        """
        Given a list of arguments, return pairwise relations: support, attack, indifferent.
        Returns a dict where keys are "i-j" indices and values are the relation.
        """
        from prompts import ARGUMENT_RELATION_PROMPT
        prompt = ARGUMENT_RELATION_PROMPT.format(arguments=json.dumps(arguments))
        raw_response = self.llm.run_inference(prompt)

        try:
            relations = json.loads(raw_response)
            if isinstance(relations, dict):
                return relations
            else:
                return {"0-0": str(relations)}
        except Exception as e:
            print(f"[WARN] Failed to parse JSON: {e}")
            return {"error": raw_response}
