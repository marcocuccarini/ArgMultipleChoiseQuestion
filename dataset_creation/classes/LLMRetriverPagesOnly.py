# -----------------------------
# Wikipedia Retriever using LLM
# -----------------------------

import json
from PromptBuilder import PromptBuilder  # make sure PromptBuilder.py is in the same folder

class LLMRetrieverPagesOnly:
    """
    A simple retrieval class that uses the LLM to suggest relevant Wikipedia pages.
    """
    def __init__(self, llm):
        self.llm = llm

    def get_candidate_pages(self, question, choices, max_pages=5):
        """
        Generate a list of relevant Wikipedia page titles using the LLM.
        """
        # Build prompt using the PromptBuilder
        prompt = PromptBuilder.wikipedia_retrieval_prompt(question, choices, max_pages)
        
        # Send prompt to LLM
        response = self.llm.send_prompt(prompt)
        
        try:
            # Try parsing LLM response as JSON
            pages = json.loads(response.raw_text)
            if isinstance(pages, list):
                return pages
        except Exception:
            pass  # fallback if response is not valid JSON
        
        # Return raw text as fallback
        return [response.raw_text]
