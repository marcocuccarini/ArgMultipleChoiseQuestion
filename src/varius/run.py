import json

with open("dataset/sciq_top20_intervals_with_facts.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

ranked_evidence = []

for entry in dataset:
    evidence_entry = {}
    for interval in entry.get("top_intervals", []):
        page = interval.get("page_title", "unknown")
        text = interval.get("interval", "")
        if page not in evidence_entry:
            evidence_entry[page] = []
        evidence_entry[page].append({"text": text})
    ranked_evidence.append(evidence_entry)

with open("ranked_evidence.json", "w", encoding="utf-8") as f:
    json.dump(ranked_evidence, f, indent=2, ensure_ascii=False)
