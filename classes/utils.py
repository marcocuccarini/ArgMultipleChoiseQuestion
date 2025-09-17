# utils.py

import re

def clean_json_response(raw_response: str) -> str:
    """
    Remove Markdown code blocks (```json ... ```) and extra spaces.
    """
    cleaned = re.sub(r"```(?:json)?\n(.*?)```", r"\1", raw_response, flags=re.DOTALL)
    return cleaned.strip()
