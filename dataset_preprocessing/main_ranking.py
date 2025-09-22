import json
import re
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # ✅ add tqdm for progress bar

# ----------------
# Load model
# ----------------
print("⏳ Loading model...")
model_name = "facebook/contriever"  # unsupervised dense retriever
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model loaded")

def encode(texts, desc="Encoding"):
    """Encode texts into dense vectors with mean pooling and tqdm progress."""
    embeddings_list = []
    for batch in tqdm(texts, desc=desc, unit="text"):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)
        embeddings_list.append(emb)
    return torch.cat(embeddings_list, dim=0)

def chunk_text(text):
    """Split text into sentence-like chunks using punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def process_quiz(item):
    """
    For a single quiz item:
      - split evidence into chunks
      - rank all chunks against each fact
      - replace facts_per_choice with sorted list of evidence chunks
    """
    # Step 1: Build evidence chunks
    evidence_chunks = []
    for choice, evidence in item["evidence_per_choice"].items():
        for idx, chunk in enumerate(chunk_text(evidence)):
            evidence_chunks.append({
                "choice": choice,
                "chunk_id": f"{choice}_chunk{idx+1}",
                "text": chunk
            })
    
    # Step 2: Encode all chunks
    chunk_embeddings = encode([c["text"] for c in evidence_chunks], desc="Encoding evidence")

    # Step 3: Rank evidence for each fact
    ranked_facts = {}
    for choice, fact in tqdm(item["facts_per_choice"].items(), desc="Ranking facts", unit="fact"):
        fact_emb = encode([fact], desc=f"Encoding fact: {choice}")
        sims = torch.nn.functional.cosine_similarity(fact_emb, chunk_embeddings)

        # Sort by similarity
        scores = [
            {"chunk_id": evidence_chunks[i]["chunk_id"], 
             "text": evidence_chunks[i]["text"], 
             "score": float(sims[i])}
            for i in range(len(evidence_chunks))
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)

        # Save only sorted chunk texts
        ranked_facts[choice] = [s["text"] for s in scores]

    # Step 4: Replace facts_per_choice with ranked chunk lists
    item["facts_per_choice"] = ranked_facts
    return item

# ----------------
# Main
# ----------------
if __name__ == "__main__":
    input_file = "sciq_facts.json"       # Your input JSON file
    output_file = "sciq_facts_rank.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle single-item or list of items with tqdm
    if isinstance(data, dict):
        processed = process_quiz(data)
    elif isinstance(data, list):
        processed = [process_quiz(item) for item in tqdm(data, desc="Processing quizzes", unit="quiz")]
    else:
        raise ValueError("Input JSON must be a dict or list of dicts")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed file saved to {output_file}")
