from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

model_path = r"../local_models/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

#加载我们自己训练的权重（中文古诗词）
trained_parameter_path = r"../local_params/net.pt"
loaded_object = torch.load(trained_parameter_path, map_location=torch.device('cpu'))
model.load_state_dict(loaded_object)

#使用系统自带的pipeline工具生成内容
pipeline = TextGenerationPipeline(model, tokenizer, device=0)

print(pipeline("天高", max_length =24))
# [{'generated_text': '天高, 地 僻 不 可 居. 山 色 连 云 起, 溪 声 入 竹 余'}]