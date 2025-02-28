import requests
#使用Token访问在线模型

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(API_URL,headers=headers,json={"inputs":"你好，Hugging face"})
print(response.json())