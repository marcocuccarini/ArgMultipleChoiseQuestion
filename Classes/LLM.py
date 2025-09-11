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

    def run_inference(self, prompt: str) -> str:
        try:
            available = [m["name"] for m in ollama.list()["models"]]
            if self.model not in available:
                print(f"[INFO] Model '{self.model}' not found locally. Pulling...")
                ollama.pull(self.model)

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a reasoning assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": self.temperature, "num_predict": self.max_tokens}
            )
            return response["message"]["content"]

        except Exception as e:
            print(f"[ERROR] Ollama inference failed: {e}")
            return json.dumps({"error": str(e)})
