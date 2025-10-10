# file: dataset_creation/InformationRetrieval_DiverseContrastive_Top20_WithFacts.py

import json
import wikipedia
from sentence_transformers import SentenceTransformer, util
import torch
import os
import time
from requests.exceptions import RequestException

# ----------------------------
# Load JSON data
# ----------------------------
def load_question_file(file_path):
    """Carica un file JSON e ritorna la lista dei record."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' non trovato.")
        return None
    except json.JSONDecodeError:
        print(f"File '{file_path}' non è un JSON valido.")
        return None

# ----------------------------
# Fetch Wikipedia Pages con retry e timeout
# ----------------------------
def fetch_wikipedia_pages(page_titles, max_retries=3, sleep_between=2):
    """
    Scarica il contenuto di pagine Wikipedia da una lista di titoli,
    con retry e gestione degli errori.
    Ritorna un dizionario {title: content}.
    """
    pages_content = {}
    for title in page_titles:
        for attempt in range(max_retries):
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
                pages_content[title] = page.content
                break
            except wikipedia.DisambiguationError as e:
                print(f"Disambiguation error per '{title}': scelgo '{e.options[0]}'")
                try:
                    page = wikipedia.page(e.options[0])
                    pages_content[title] = page.content
                    break
                except Exception as e2:
                    print(f"Fallito fetch prima opzione disambiguation: {e2}")
                    pages_content[title] = ""
                    break
            except wikipedia.PageError:
                print(f"Pagina '{title}' non trovata.")
                pages_content[title] = ""
                break
            except RequestException as e:
                print(f"Errore di rete per '{title}' (tentativo {attempt+1}/{max_retries}): {e}")
                time.sleep(sleep_between)
            except Exception as e:
                print(f"Errore inatteso per '{title}': {e}")
                pages_content[title] = ""
                break
        else:
            print(f"Fallito fetch '{title}' dopo {max_retries} tentativi.")
            pages_content[title] = ""
    return pages_content

# ----------------------------
# Utility: split text into chunks con overlap
# ----------------------------
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Divide un testo in chunk di lunghezza fissa con overlap.
    Ogni chunk ha al massimo 'chunk_size' caratteri.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size deve essere maggiore di overlap")

    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start += (chunk_size - overlap)
    
    return chunks

# ----------------------------
# Contrastive-diverse interval ranking
# ----------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # carica una volta sola

def rank_intervals_contrastive_diverse(question, pages_content, top_k=20, diversity_lambda=0.7, chunk_size=500, overlap=50):
    """
    Ritorna i migliori intervalli usando contrastive-diverse scoring:
    - Rilevante per la domanda
    - Diverso rispetto agli intervalli già selezionati
    Ritorna lista di tuple: (interval_text, score, page_title)
    """
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    all_intervals = []
    interval_titles = []
    for title, content in pages_content.items():
        chunks = split_text_into_chunks(content, chunk_size=chunk_size, overlap=overlap)
        all_intervals.extend(chunks)
        interval_titles.extend([title]*len(chunks))

    if not all_intervals:
        return []

    interval_embeddings = embedding_model.encode(all_intervals, convert_to_tensor=True)
    sim_to_question = util.cos_sim(question_embedding, interval_embeddings)[0].clone().detach()

    selected = []

    for _ in range(min(top_k, len(all_intervals))):
        best_idx = torch.argmax(sim_to_question).item()
        selected.append((all_intervals[best_idx], sim_to_question[best_idx].item(), interval_titles[best_idx]))

        sim_selected = util.cos_sim(interval_embeddings, interval_embeddings[best_idx])
        sim_to_question -= diversity_lambda * sim_selected.squeeze()
        sim_to_question[best_idx] = -float('inf')

    return selected

# ----------------------------
# Main function
# ----------------------------
def main():
    file_path = "dataset/sciq_with_wiki.json"
    output_file = "dataset/sciq_top20_intervals_with_facts.json"

    data = load_question_file(file_path)
    if not data:
        return

    # Carica eventuale output esistente
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        processed_ids = {rec["id"] for rec in output_data}
        print(f"Riprendo da file esistente. {len(processed_ids)} domande già processate.")
    else:
        output_data = []
        processed_ids = set()

    for idx, record in enumerate(data):
        if record.get("id") in processed_ids:
            continue

        retrieved_pages = record.get("retrieved_pages", [])
        if not retrieved_pages:
            print(f"Nessuna retrieved_pages per domanda id {record.get('id')}.")
            continue

        # Pulizia retrieved_pages (rimuove ```json```)
        clean_pages = []
        for page in retrieved_pages:
            page = page.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(page)
                if isinstance(parsed, list):
                    clean_pages.extend(parsed)
            except json.JSONDecodeError:
                clean_pages.append(page)

        # Fetch page content
        pages_content = fetch_wikipedia_pages(clean_pages)

        # Rank intervals
        ranked_intervals = rank_intervals_contrastive_diverse(
            record.get("question"),
            pages_content,
            top_k=20,
            chunk_size=500,
            overlap=50
        )

        # Salva risultati
        output_data.append({
            "id": record.get("id"),
            "question": record.get("question"),
            "choices": record.get("choices"),
            "correct_answer": record.get("correct_answer"),
            "choice_facts": record.get("choice_facts", {}),
            "top_intervals": [
                {"interval": p, "score": score, "page_title": title}
                for p, score, title in ranked_intervals
            ]
        })

        # Anteprima top interval
        print(f"\n=== Question ID {record.get('id')} ===")
        print(f"Question: {record.get('question')}\n")
        for i, item in enumerate(output_data[-1]["top_intervals"], start=1):
            print(f"--- Rank {i} (Score={item['score']:.3f}, Page={item['page_title']}) ---\n{item['interval'][:300]}...\n")

        # Salvataggio periodico ogni 5 domande
        if (len(output_data) % 5 == 0) or (idx == len(data)-1):
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Progress salvato dopo {len(output_data)} domande.")

    print(f"✅ Top 20 intervals con facts salvati in '{output_file}'")

if __name__ == "__main__":
    main()
