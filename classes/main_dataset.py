import os
import json
from classes.dataset_preparer import SciQDatasetPreparer

# -----------------------------
# Use local sample dataset
# -----------------------------
column_oriented_dataset = {
    "question": [
        "Compounds that are capable of accepting electrons, such as O2 or F2, are called what?",
        "What term in biotechnology means a genetically exact copy of an organism?",
        "Vertebrata are characterized by the presence of what?",
        "What is the height above or below sea level called?",
        "Ice cores, varves and what else indicate the environmental conditions at the time of their creation?"
    ],
    "distractor1": ["antioxidants", "adult", "Bones", "depth", "mountain ranges"],
    "distractor2": ["Oxygen", "male", "Muscles", "latitude", "fossils"],
    "distractor3": ["residues", "phenotype", "Thumbs", "variation", "magma"],
    "correct_answer": ["oxidants", "clone", "backbone", "elevation", "tree rings"],
    "support": [
        "Oxidants and Reductants Compounds that are capable of accepting electrons...",
        "But transgenic animals just have one novel gene. Could a clone be developed...",
        "Figure 29.7 Vertebrata are characterized by the presence of a backbone...",
        "As you know, the surface of Earth is not flat. Elevation is the height above or below sea level...",
        "Tree rings, ice cores, and varves indicate the environmental conditions at the time they were made."
    ]
}

# -----------------------------
# Initialize the dataset preparer
# -----------------------------
prep = SciQDatasetPreparer(
    split="test",                 # only used if HuggingFace dataset is loaded
    model="mistral",              # your LLM model
    dataset=column_oriented_dataset if use_local_dataset else None,
    use_llm_user=True,            # enable fact verbalization
    use_ir=True                   # enable Wikipedia evidence retrieval
)

# -----------------------------
# Prepare records
# -----------------------------
records = prep.prepare_records(max_examples=5)  # adjust number as needed

# -----------------------------
# Save to JSON
# -----------------------------
output_path = os.path.join(os.getcwd(), "sciq_facts.json")
prep.save_to_json(records, path=output_path)

print(f"✅ Dataset prepared and saved to {output_path}")
