import requests
import dotenv
import os

dotenv.load_dotenv()

HF_API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
HF_TOKEN=os.getenv("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
# No token required, for annoymous access

response = requests.post(HF_API_URL, headers=headers, json={"inputs": "你好，我是一个AI"})

print(response.json())