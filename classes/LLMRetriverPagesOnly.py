# -----------------------------
# Wikipedia Retriever using LLM
# -----------------------------
class LLMRetrieverPagesOnly:
    def __init__(self, llm):
        self.llm = llm

    def get_candidate_pages(self, question, choices, max_pages=5):
        prompt = PromptBuilder.wikipedia_retrieval_prompt(question, choices, max_pages)
        response = self.llm.send_prompt(prompt)
        try:
            pages = json.loads(response.raw_text)
            if isinstance(pages, list):
                return pages
        except Exception:
            pass
        return [response.raw_text]
