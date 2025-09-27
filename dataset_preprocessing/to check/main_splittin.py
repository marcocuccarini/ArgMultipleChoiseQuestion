import json

# -----------------------------
# Extract only facts_per_choice
# -----------------------------
def extract_facts(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If the JSON is a list of examples
    if isinstance(data, list):
        extracted = [entry.get("facts_per_choice", {}) for entry in data]
    else:  # If it's just one dictionary
        extracted = data.get("facts_per_choice", {})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"✅ Extracted facts saved to {output_file}")


# Example usage
if __name__ == "__main__":
    extract_facts("sciq_facts.json", "facts_only.json")
