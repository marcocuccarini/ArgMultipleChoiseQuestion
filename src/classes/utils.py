# utils.py
import wikipedia
import json
import re
import time
from requests.exceptions import RequestException

def clean_json_response(raw_response: str) -> str:
    """
    Remove Markdown code blocks (```json ... ```) and extra spaces.
    """
    cleaned = re.sub(r"```(?:json)?\n(.*?)```", r"\1", raw_response, flags=re.DOTALL)
    return cleaned.strip()

def load_question_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore nel caricamento file {file_path}: {e}")
        return None

def resume_file(output_file):

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        downloaded_titles = set(existing.keys())
        print(f"Riprendo da file esistente: {len(downloaded_titles)} pagine già scaricate.")
    else:
        existing = {}
        downloaded_titles = set()

    return downloaded_titles



def section_extraction(page):
    """
    Estrae le sezioni principali da un oggetto `wikipedia.page` e 
    restituisce un dizionario organizzato come:
    
    {
        "TitoloPagina": {
            "Introduzione": "testo prima della prima sezione",
            "Sezione1": "testo...",
            "Sezione2": "testo...",
            ...
        }
    }
    """
    content = page.content
    title = page.title

    # Divide content into sections (== Section ==)
    parts = re.split(r'\n==+\s*(.*?)\s*==+\n', content)

    sezioni = {}

    # 🟦 Add the introduction if present (text before first section)
    intro_text = parts[0].strip()
    if intro_text:
        sezioni["Introduzione"] = intro_text

    # 🟩 Add all other sections
    for i in range(1, len(parts), 2):
        
        sezione = parts[i].strip()

        if (sezione not in ["See also","References","External links"]):

            testo = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sezioni[sezione] = testo

    return sezioni





# ----------------------------
# Fetch Wikipedia Pages with retry
# ----------------------------
def fetch_wikipedia_pages(page_titles, max_retries=3, sleep_between=2):
    pages_content = {}

    for title in page_titles:
        for attempt in range(max_retries):
            try:
                page = wikipedia.page(title[0], auto_suggest=False, redirect=True)
                sections = section_extraction(page)
                # 🔹 Split each section into paragraphs
                for section_name, text in sections.items():
                    sections[section_name] = text.split("\n\n")
                pages_content[title[0]] = sections
                break

            except wikipedia.DisambiguationError as e:
                print(f"Disambiguation per '{title[0]}', scelgo '{e.options[0]}'")
                try:
                    page = wikipedia.page(e.options[0])
                    sections = section_extraction(page)
                    for section_name, text in sections.items():
                        sections[section_name] = text.split("\n\n")
                    pages_content[title[0]] = sections
                    break
                except Exception:
                    pages_content[title[0]] = {}
                    break
            except wikipedia.PageError:
                print(f"Pagina '{title[0]}' non trovata.")
                pages_content[title[0]] = {}
                break
            except RequestException as e:
                print(f"Errore di rete ({attempt+1}/{max_retries}) per '{title[0]}': {e}")
                time.sleep(sleep_between)
            except Exception as e:
                print(f"Errore inatteso per '{title[0]}': {e}")
                pages_content[title[0]] = {}
                break
        else:
            print(f"Fallito fetch '{title[0]}' dopo {max_retries} tentativi.")
            pages_content[title[0]] = {}

    return pages_content
