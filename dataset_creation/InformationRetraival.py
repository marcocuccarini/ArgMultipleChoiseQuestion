import json
import wikipedia

def load_question_file(file_path):
    """Load a JSON file and return its content as a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"File '{file_path}' is not valid JSON.")
        return None

def fetch_wikipedia_pages(page_titles):
    """Download Wikipedia page content for a list of page titles."""
    pages_content = {}
    for title in page_titles:
        try:
            page = wikipedia.page(title)
            pages_content[title] = page.content
        except wikipedia.DisambiguationError as e:
            print(f"Disambiguation error for '{title}': choosing first option '{e.options[0]}'")
            pages_content[title] = wikipedia.page(e.options[0]).content
        except wikipedia.PageError:
            print(f"Page '{title}' not found.")
            pages_content[title] = ""
    return pages_content

def main():
    file_path = input("Enter path to JSON file: ").strip()
    data = load_question_file(file_path)
    if not data:
        return

    # Extract retrieved pages
    retrieved_pages = data.get("retrieved_pages", [])
    if not retrieved_pages:
        print("No retrieved_pages found in the JSON.")
        return

    # Fetch page contents
    pages_content = fetch_wikipedia_pages(retrieved_pages)

    # Display results
    for title, content in pages_content.items():
        print(f"\n--- Wikipedia page: {title} ---\n")
        print(content[:1000])  # Print first 1000 characters for preview

if __name__ == "__main__":
    main()
