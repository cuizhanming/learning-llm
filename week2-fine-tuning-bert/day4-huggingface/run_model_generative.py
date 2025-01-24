from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from load_model import download_model

# Load the model and tokenizer from the local directory
# model_dir="./local_models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

model_id = "uer/gpt2-chinese-cluecorpussmall"
model_dir = download_model(model_name=model_id)

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device="cpu")
# Generate text using the model
output = generator(
    "你好，你从哪里来？",
    max_length=50,
    num_return_sequences=2,
    truncation=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    clean_up_tokenization_spaces=False
)

print(output)