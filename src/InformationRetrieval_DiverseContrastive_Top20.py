import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Embedding model
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------
# Rank paragraphs per page with Introduzione separate
# ----------------------------
def rank_paragraphs_by_pile(question, wiki_data, top_k=10, diversity_lambda=0.7):
    """
    Rank paragraphs per page, with 'Introduzione' ranked separately.
    Other sections are ranked separately.
    """
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    page_results = {}

    for page_title, sections in wiki_data.items():
        if not isinstance(sections, dict):
            continue

        # --- Split piles ---
        summary_paragraphs = sections.get("Introduzione", [])
        other_paragraphs = []
        for sec, paras in sections.items():
            if sec == "Introduzione" or not isinstance(paras, list):
                continue
            for p in paras:
                if p.strip():
                    other_paragraphs.append((p.strip(), sec))

        # --- Rank summary paragraphs ---
        summary_ranked = []
        if summary_paragraphs:
            summary_emb = embedding_model.encode(summary_paragraphs, convert_to_tensor=True)
            sim_summary = util.cos_sim(question_embedding, summary_emb)[0]
            sim_copy = sim_summary.clone().detach()
            for _ in range(min(top_k, len(summary_paragraphs))):
                best_idx = torch.argmax(sim_copy).item()
                summary_ranked.append({
                    "text": summary_paragraphs[best_idx],
                    "score": sim_summary[best_idx].item(),
                    "section": "Introduzione"
                })
                # diversity penalty
                sim_selected = util.cos_sim(summary_emb, summary_emb[best_idx])
                sim_copy -= diversity_lambda * sim_selected.squeeze()
                sim_copy[best_idx] = -float("inf")

        # --- Rank other paragraphs ---
        other_ranked = []
        if other_paragraphs:
            texts = [p[0] for p in other_paragraphs]
            para_emb = embedding_model.encode(texts, convert_to_tensor=True)
            sim_other = util.cos_sim(question_embedding, para_emb)[0]
            sim_copy = sim_other.clone().detach()
            for _ in range(min(top_k, len(other_paragraphs))):
                best_idx = torch.argmax(sim_copy).item()
                other_ranked.append({
                    "text": other_paragraphs[best_idx][0],
                    "score": sim_other[best_idx].item(),
                    "section": other_paragraphs[best_idx][1]
                })
                sim_selected = util.cos_sim(para_emb, para_emb[best_idx])
                sim_copy -= diversity_lambda * sim_selected.squeeze()
                sim_copy[best_idx] = -float("inf")

        if summary_ranked or other_ranked:
            page_results[page_title] = {
                "summary_ranked": summary_ranked,
                "other_ranked": other_ranked
            }

    return page_results


# ----------------------------
# Process single question item
# ----------------------------
def process_question_item(item):
    q_field = item.get("question")

    # Convert question to text
    if isinstance(q_field, list):
        question = " ".join([q[0] for q in q_field])
    elif isinstance(q_field, str):
        question = q_field
    else:
        raise ValueError(f"Invalid question format: {type(q_field)}")

    wiki_data = item.get("retrieved_page", {})
    if not wiki_data:
        print(f"⚠️ Skipping ID {item.get('id', '?')} — no retrieved_page")
        return None

    ranked_pages = rank_paragraphs_by_pile(question, wiki_data, top_k=20)
    item["ranked_pages"] = ranked_pages
    return item


# ----------------------------
# Main script
# ----------------------------
def main():
    input_file = "dataset/wiki_pages_content.json"
    output_file = "dataset/wiki_ranked_pages.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load input JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    if isinstance(data, list):
        for item in data:
            processed = process_question_item(item)
            if processed:
                results.append(processed)
    elif isinstance(data, dict):
        processed = process_question_item(data)
        if processed:
            results.append(processed)
    else:
        raise ValueError("JSON must be a list or dict")

    # Save output JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved ranked pages for {len(results)} questions to '{output_file}'")


if __name__ == "__main__":
    main()
