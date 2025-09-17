import sys
import os
import json

# Add classes folder to module path
sys.path.append(os.path.join(os.path.dirname(__file__), "../classes"))

from dataset_preparer import SciQDatasetPreparer

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

# --- Pass dataset into preparer ---
prep = SciQDatasetPreparer(
    dataset=column_oriented_dataset,  # column-oriented dict
    model="gemma3:27b",
    use_llm_user=True,
    use_ir=False  # no Wikipedia IR
)

records = prep.prepare_records(max_examples=4)
prep.save_to_json(records, "sciq_facts.json")

# Debugging: check one example
print(json.dumps(records[0], indent=2))
