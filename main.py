from llm import LLM
from llm_user import LLMUser

# Instantiate low-level LLM
llm_model = LLM(model="mistral")

# Create high-level user instance
user = LLMUser(llm_model)

text = "We should ban smoking in public places because it harms others. However, some argue this violates personal freedom."

# Step 1: Extract arguments
arguments = user.extract_arguments_with_ollama(text)
print("Arguments:", arguments)

# Step 2: Detect relations
relations = user.detect_argument_relations(arguments)
print("Relations:", relations)
    