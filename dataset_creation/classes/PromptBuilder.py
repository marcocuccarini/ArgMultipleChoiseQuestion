import json

class PromptBuilder:
    """
    Builds and manages prompts for different LLM tasks.
    Each method returns a formatted text prompt that can be
    sent to an LLM (e.g. OllamaChat).
    """

    @staticmethod
    def wikipedia_retrieval_prompt(question: str, choices: list, max_pages: int = 5) -> str:
        """
        Build a prompt for retrieving relevant Wikipedia pages.
        """
        return f"""
You are a retrieval model.
Given the question and its multiple-choice options below,
suggest up to {max_pages} Wikipedia page titles that are most relevant
for finding the correct answer.

Question:
{question}

Choices:
{json.dumps(choices, ensure_ascii=False, indent=2)}

Respond ONLY with a JSON list of Wikipedia page titles, e.g.:
["Photosynthesis", "Light-dependent reactions", "Chlorophyll"]
"""

    @staticmethod
    def explanation_prompt(question: str, correct_answer: str, retrieved_pages: list) -> str:
        """
        Build a prompt for generating an explanation based on retrieved Wikipedia pages.
        """
        pages_text = ", ".join(retrieved_pages)
        return f"""
You are an expert science tutor.
Explain why the correct answer "{correct_answer}" is true for the question below,
using knowledge from the following Wikipedia pages: {pages_text}.

Question:
{question}

Your explanation should be clear, concise, and factual.
"""

    @staticmethod
    def fact_alignment_prompt(facts: dict, choices: list) -> str:
        """
        Build a prompt for aligning extracted facts with each answer choice.
        """
        return f"""
You are a reasoning assistant.
Given these facts:
{json.dumps(facts, ensure_ascii=False, indent=2)}

And these choices:
{json.dumps(choices, ensure_ascii=False, indent=2)}

For each choice, indicate which facts are relevant and why.
Respond in JSON as:
{{
  "choice1": ["fact1", "fact3"],
  "choice2": []
}}
"""
