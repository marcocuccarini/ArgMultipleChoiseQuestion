import re
import json

def split_into_sections(text: str) -> dict:
    """Split Wikipedia-style text into a dict of {section_title: content}."""
    sections = {}
    pattern = re.compile(r"(={2,})\s*(.*?)\s*\1")  # matches == Title == or === Title ===
    matches = list(pattern.finditer(text))

    if not matches:  # no sections, everything is intro
        return {"Introduction": text.strip()}

    intro = text[:matches[0].start()].strip()
    if intro:
        sections["Introduction"] = intro

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections[title] = content

    return sections


def process_entry(entry: dict) -> dict:
    """Process a single question entry."""
    entry["evidence_per_choice"] = {
        k: split_into_sections(v) for k, v in entry["evidence_per_choice"].items()
    }
    return entry


def process_dataset(input_path: str, output_path: str):
    """Read JSON file, process all entries, and save to output."""
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    processed = [process_entry(entry) for entry in dataset]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)


# Example usage
if __name__ == "__main__":
    process_dataset("sciq_facts.json", "sciq_facts_output.json")
