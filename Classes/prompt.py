# prompts.py

ARGUMENT_EXTRACTION_PROMPT = """
Extract all argumentative statements (claims or premises) from the text below.
Return the result strictly as a JSON list of strings, without explanations.

Text:
\"\"\"{text}\"\"\"

Example:
Input: "We should ban smoking because it harms others. However, some argue it violates freedom."
Output: ["We should ban smoking", "It harms others", "It violates freedom"]

Now extract the arguments from the given text.
"""

# 🔹 Old prompt kept if needed
ARGUMENT_RELATION_PROMPT = """
You are an argumentation reasoning assistant. 
Given a list of arguments, determine the relationship between each pair:
- "support": the first argument supports the second,
- "attack": the first argument attacks the second,
- "indifferent": neither attack nor support.

Return a JSON dictionary where keys are pairs of indices and values are the relation.

Arguments:
{arguments}
"""

# 🔹 New pairwise relation prompt
PAIRWISE_RELATION_PROMPT = """
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
