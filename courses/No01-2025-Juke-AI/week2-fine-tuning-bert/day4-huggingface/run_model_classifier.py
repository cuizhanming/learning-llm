from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load the previously downloaded model
# model_dir = "./local_models/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
# model = BertForSequenceClassification.from_pretrained(model_dir)
# tokenizer = BertTokenizer.from_pretrained(model_dir)
# # print(model)
#
# # Instantiate a text classification pipeline
# classifier = pipeline(task='sentiment-analysis', model=model, tokenizer=tokenizer, device="cpu")
#
# # Perform classification
# res = classifier("I love you")
# print(res)

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

def download_and_create_classifier(task, model, cache_dir="local_models", device="cpu"):
    # Load the model and tokenizer
    cache_dir = f"{cache_dir}/{model}"
    model = AutoModelForTokenClassification.from_pretrained(model, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    # Instantiate a text classification pipeline
    return pipeline(task=task, model=model, tokenizer=tokenizer, device=device)

classifier = download_and_create_classifier(
    task = "token-classification",
	model="google-bert/bert-base-chinese",
    cache_dir="../../local_models"
)

print(classifier("The text to be classified."))
