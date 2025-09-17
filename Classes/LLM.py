# llm.py

import json
import ollama

class LLM:
    """
    Low-level wrapper for Ollama LLM.
    Handles model loading and raw prompt inference.
    """

    def __init__(self, model: str = "mistral", temperature: float = 0.7, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Check if model exists locally, pull if not
        try:
            available = [m["name"] for m in ollama.list()["models"]]
            if self.model not in available:
                print(f"[INFO] Model '{self.model}' not found locally. Pulling...")
                ollama.pull(self.model)
        except Exception as e:
            print(f"[WARN] Could not verify local models: {e}")


    def extract_arguments_with_ollama(self, text: str) -> list:
        prompt = ARGUMENT_EXTRACTION_PROMPT.format(text=text)
        raw_response = self.llm.run_inference(prompt)
        try:
            cleaned = clean_json_response(raw_response)
            arguments = json.loads(cleaned)
            return arguments if isinstance(arguments, list) else [str(arguments)]
        except Exception as e:
            print(f"[WARN] Failed to parse JSON from argument extraction: {e}")
            return [raw_response]

    def detect_argument_relations(self, arguments: list) -> dict:
        prompt = ARGUMENT_RELATION_PROMPT.format(arguments=json.dumps(arguments))
        raw_response = self.llm.run_inference(prompt)
        try:
            cleaned = clean_json_response(raw_response)
            relations = json.loads(cleaned)
            return relations if isinstance(relations, dict) else {"0-0": str(relations)}
        except Exception as e:
            print(f"[WARN] Failed to parse JSON from relation detection: {e}")
            return {"error": raw_response}




    def run_inference(self, prompt: str) -> str:
        """
        Run a prompt through the model and return raw text.
        Handles errors and ensures valid JSON output.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a reasoning assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            # Ollama responses include a "message" dict with "content"
            return response["message"]["content"]
        except Exception as e:
            print(f"[ERROR] Ollama inference failed: {e}")
            return json.dumps({"error": str(e)})
