# classes/LLM.py
from classes.classes_server_ollama import OllamaServer, OllamaChat, LLMResponse
import datetime

class LLM:
    def __init__(self, model: str):
        self.server = OllamaServer()
        self.model = model
        self.chat = OllamaChat(self.server, model)

    def run_inference(self, prompt: str) -> str:
        response = self.chat.send_prompt(prompt)
        if response.response_type == "generated":
            return response.raw_text
        else:
            raise RuntimeError(f"LLM Error: {response.raw_text}")
