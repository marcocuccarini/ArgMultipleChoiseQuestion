import json

class LLMRetrieverPagesOnly:
    """
    Step 1 & 2 of retrieval:
    - Input: question + all candidate answers
    - Output: list of Wikipedia page titles relevant to the question
    """

    def __init__(self, llm):
        """
        llm: an instance of your LLM wrapper (e.g., classes.LLM)
        """
        self.llm = llm

    def get_candidate_pages(self, question: str, choices: list, max_pages: int = 5) -> list:
        """
        Ask the LLM to suggest relevant Wikipedia pages for the question.
        Returns a list of page titles (up to max_pages).
        """
        prompt = f"""
        You are a retrieval assistant.
        Given a multiple-choice question and its answer options,
        suggest up to {max_pages} Wikipedia page titles that are most likely
        to contain information needed to answer the question.

        Return ONLY a JSON list of page titles.

        Question: "{question}"
        Options: {json.dumps(choices)}
        """

        raw = self.llm.run_inference(prompt)

        try:
            pages = json.loads(raw)
            if isinstance(pages, list):
                return pages[:max_pages]
            return []
        except Exception as e:
            print(f"[WARN] Failed to parse candidate pages JSON: {e}")
            return []
