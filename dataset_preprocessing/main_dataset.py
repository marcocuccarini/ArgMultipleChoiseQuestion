import sys
import os
import json

# Add classes folder to module path
sys.path.append(os.path.join(os.path.dirname(__file__), "../classes"))

from dataset_preparer import SciQDatasetPreparer

# Load column-oriented JSON


from datasets import load_dataset

# Load train/validation/test splits
dataset = load_dataset("allenai/sciq")

# Convert column-oriented → list of records expected by preparer


# Pass into SciQDatasetPreparer
prep = SciQDatasetPreparer(
    dataset=dataset["test"],
    model="gemma3:27b",
    use_llm_user=True,
    use_ir=True,
)

prepared_records = prep.prepare_records(max_examples=1000)
prep.save_to_json(prepared_records, "sciq_facts.json")

# Check first prepared record
if prepared_records:
    print(json.dumps(prepared_records[0], indent=2))
else:
    print("⚠️ No records were prepared. Check dataset completeness.")
