import json

class PromptBuilder:
    """
    Builds and manages prompts for different LLM tasks.
    Each method returns a formatted text prompt that can be
    sent to an LLM (e.g. OllamaChat).
    """
    def fact_generation_prompt(question:str, choise:str) -> str:

        return f"""
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

    def wikipedia_retrieval_prompt(question: str, choices: list, max_pages: int = 5) -> str:
        """
        Build a prompt for retrieving relevant Wikipedia pages.
        """
        return f"""
        You are a retrieval model.
        Given the question and its multiple-choice options below,
        suggest up to {max_pages} Wikipedia page titles that are most relevant
        for finding the correct answer.
        For each page, also provide a relevance value between 0 and 1.

        Question:
        {question}

        Choices:
        {json.dumps(choices, ensure_ascii=False, indent=2)}

        Respond ONLY with a JSON list of Wikipedia page titles, e.g.:
        [["Photosynthesis",0.7], ["Light-dependent reactions",0.4], ["Chlorophyll",0.3]]
        """

    def argument_extraction_prompt(text: str) -> str:

        return f"""
        Extract all argumentative statements (claims or premises) from the text below.
        Return the result strictly as a JSON list of strings, without explanations.

        Text:
        \"\"\"{text}\"\"\"

        Example:
        Input: "We should ban smoking because it harms others. However, some argue it violates freedom."
        Output: ["We should ban smoking", "It harms others", "It violates freedom"]

        Now extract the arguments from the given text.
        """

        
    def pairwise_relation_prompt(arg_a: str, arg_b:str) -> str:

        return f"""
        You are an argumentation reasoning assistant. 
        Compare the following two arguments:

        Argument A: "{arg_a}"
        Argument B: "{arg_b}"

        Decide the relation of A toward B:
        - "support": A supports B
        - "attack": A attacks B
        - "indifferent": neither support nor attack

        Return strictly as JSON: {{"relation": "<support|attack|indifferent>"}}.
        """

