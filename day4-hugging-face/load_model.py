from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model(model_name, cache_dir="./local_models"):
    try:
        # Create the cache directory if it doesn't exist
        cache_dir = os.path.join(cache_dir, model_name)
        os.makedirs(cache_dir, exist_ok=True)

        # Load the model and tokenizer
        AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        # Get the absolute path of the cache directory
        absolute_cache_dir = os.path.abspath(cache_dir)

        # Get the snapshot path
        snapshot_path = os.path.join(absolute_cache_dir, "models--"+model_name.replace("/", "--"), "snapshots")
        snapshot_id = os.listdir(snapshot_path)[0]  # Get the first snapshot id
        snapshot_path = os.path.join(snapshot_path, snapshot_id)

        return snapshot_path
    except Exception as e:
        print("An error occurred while downloading the model.")
        print(e)
        return None

# generator models
# model_name = "uer/gpt2-chinese-cluecorpussmall"

# classifier models
# model_name = "uer/roberta-base-finetuned-cluener2020-chinese"
model_name = "google-bert/bert-base-chinese"

snapshot_path = download_model(model_name)
print(f"Model {model_name} loaded successfully! Snapshot path: {snapshot_path}")

# BertTokenizer
# 21128