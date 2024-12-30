from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-chinese"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
