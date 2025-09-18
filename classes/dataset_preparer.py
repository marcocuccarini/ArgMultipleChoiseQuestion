import json
from tqdm import tqdm
from LLM import LLM
from LLMUser import LLMUser
import wikipedia
import warnings
from bs4 import BeautifulSoup
import difflib

# -----------------------------
# Patch BeautifulSoup in Wikipedia
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
_original_bs4_init = BeautifulSoup.__init__

def patched_bs4_init(self, *args, **kwargs):
    if "features" not in kwargs:
        kwargs["features"] = "lxml"
    _original_bs4_init(self, *args, **kwargs)

BeautifulSoup.__init__ = patched_bs4_init


class SciQDatasetPreparer:
    """
    SciQ dataset preparation with:
    - Wikipedia evidence retrieval per choice
    - LLM fact verbalization per choice (choice as sentence subject)
    """

    def __init__(self, split="test", model="mistral", dataset=None, use_llm_user=True, use_ir=True):
        self.dataset = dataset.to_dict()
        self.llm = LLM(model=model)
        self.use_llm_user = use_llm_user
        self.llm_user = LLMUser(self.llm) if use_llm_user else None
        self.use_ir = use_ir
        self._wiki_cache = {}

    def get_full_evidence_for_choice(self, question: str, choice: str) -> str:
        """Retrieve relevant Wikipedia information for a given choice, with retries & fuzzy matching."""
        cache_key = f"{choice}"
        if cache_key in self._wiki_cache:
            return self._wiki_cache[cache_key]

        try:
            results = wikipedia.search(choice)
            if not results:
                self._wiki_cache[cache_key] = "No results."
                return "No results."

            # Pick the closest match by similarity to the choice
            best_match = difflib.get_close_matches(choice, results, n=1, cutoff=0.0)
            if best_match:
                target_title = best_match[0]
            else:
                target_title = results[0]  # fallback to first result

            # Try multiple candidates if first fails
            page = None
            for title in [target_title] + results[1:5]:  # check top 5 results
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    break
                except Exception:
                    continue

            if not page:
                self._wiki_cache[cache_key] = "No results."
                return "No results."

            full_text = page.content

            # Look for overlap between question & choice
            keywords = set(question.lower().split()) | set(choice.lower().split())
            relevant_paragraphs = [
                para for para in full_text.split("\n")
                if any(word in para.lower() for word in keywords)
            ]

            evidence = "\n".join(relevant_paragraphs)
            if not evidence.strip():
                evidence = full_text  # fallback to full page if no paragraph matches

            self._wiki_cache[cache_key] = evidence
            return evidence

        except Exception:
            self._wiki_cache[cache_key] = "No results."
            return "No results."

    def prepare_records(self, max_examples=None):
        """
        Prepare records with evidence and fact verbalization (choice as subject).
        Handles placeholder distractors and deduplicates questions.
        """
        records = []

        # Deduplicate questions
        seen_questions = set()
        num_examples = len(self.dataset['question'])
        if max_examples is not None:
            num_examples = min(num_examples, max_examples)

        for idx in tqdm(range(num_examples), desc="Preparing SciQ dataset"):
            question = self.dataset['question'][idx]

            if question in seen_questions:
                continue  # skip duplicates
            seen_questions.add(question)

            correct = self.dataset['correct_answer'][idx]
            distractors = [
                self.dataset['distractor1'][idx],
                self.dataset['distractor2'][idx],
                self.dataset['distractor3'][idx]
            ]
            explanation = self.dataset.get('support', [None] * num_examples)[idx]

            choices = distractors + [correct]

            choice_evidence = {}
            choice_facts = {}

            for choice in choices:
                # --- Retrieve evidence ---
                evidence = self.get_full_evidence_for_choice(question, choice) if self.use_ir else ""
                choice_evidence[choice] = evidence

                # --- Fact verbalization ---
                if self.use_llm_user:
                    fact = self.llm_user.verbalize_choice(question, choice)
                else:
                    fact = ""
                choice_facts[choice] = fact

            records.append({
                "question": question,
                "choices": choices,
                "correct_answer": correct,
                "explanation": explanation,
                "evidence_per_choice": choice_evidence,
                "facts_per_choice": choice_facts
            })

        return records

    def save_to_json(self, records, path="sciq_facts.json"):
        """Save records to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(records)} records to {path}")
