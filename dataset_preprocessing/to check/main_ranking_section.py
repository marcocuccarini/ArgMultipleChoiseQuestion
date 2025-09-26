import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F

# ----------------
# Load model
# ----------------
print("⏳ Loading model...")
model_name = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model loaded")

# ----------------
# Encoding function
# ----------------
def encode_texts(texts, batch_size=8, desc="Encoding"):
    """Encode a list of texts into embeddings (mean pooling) with batching and tqdm."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)

# ----------------
# Ranking function
# ----------------
def rank_sections(evidence_dict, facts_dict, batch_size=8):
    """
    Compute similarity between each section and its corresponding fact.
    Returns a nested dict with sorted sections and similarity scores.
    """
    results = {}
    for choice, sections in tqdm(evidence_dict.items(), desc="Processing choices"):
        section_keys = list(sections.keys())
        section_texts = list(sections.values())
        section_embeddings = encode_texts(section_texts, batch_size=batch_size, desc=f"Encoding {choice}")

        fact_text = facts_dict[choice]
        fact_embedding = encode_texts([fact_text], batch_size=1, desc=f"Encoding fact {choice}")[0]

        sims = F.cosine_similarity(fact_embedding.unsqueeze(0), section_embeddings)

        scored_sections = sorted(
            [
                {"section": k, "similarity": float(s), "text": sections[k]} 
                for k, s in zip(section_keys, sims)
            ],
            key=lambda x: x["similarity"],
            reverse=True
        )
        results[choice] = scored_sections
    return results

# ----------------
# Main
# ----------------
if __name__ == "__main__":
    input_file = "sciq_facts_section.json"      # Your input JSON
    output_file = "ranked_evidence.json"

    # Load JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ranked_results_all = []

    # Determine if it's a single dict or list of dicts
    if isinstance(data, dict):
        # Single item
        evidence_per_choice = data["evidence_per_choice"]
        facts_per_choice = data["facts_per_choice"]
        ranked_results = rank_sections(evidence_per_choice, facts_per_choice, batch_size=4)
        ranked_results_all.append(ranked_results)
    elif isinstance(data, list):
        # Multiple items
        for item in tqdm(data, desc="Processing quizzes"):
            evidence_per_choice = item["evidence_per_choice"]
            facts_per_choice = item["facts_per_choice"]
            ranked_results = rank_sections(evidence_per_choice, facts_per_choice, batch_size=4)
            ranked_results_all.append(ranked_results)
    else:
        raise ValueError("Input JSON must be a dict or list of dicts")

    # Save results to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ranked_results_all, f, indent=2, ensure_ascii=False)

    print(f"✅ Ranking saved to {output_file}")
