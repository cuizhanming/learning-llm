import requests
import dotenv
import os

dotenv.load_dotenv()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_ACCESS_TOKEN")

headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
# No token required, for annoymous access

response = requests.post(HUGGINGFACE_API_URL, headers=headers, json={"inputs": "你好，我是一个AI"})

print(response.json())