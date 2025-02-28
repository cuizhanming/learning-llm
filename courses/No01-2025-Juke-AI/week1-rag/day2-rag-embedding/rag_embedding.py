"""
# !pip install --upgrade openai
# !pip install -U python-dotenv
# !pip install numpy
"""
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from numpy import dot
from numpy.linalg import norm
import numpy as np

"""
Step 2: Build LLM API Client, Ready for Embedding API Calls
"""
_ = load_dotenv(find_dotenv(), verbose=True)  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

client = OpenAI()

def get_completion(prompt, model="gpt-4o-mini"):
    """封装 openai 接口"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content

# print(get_completion("你好", model="gpt-4o-mini"))

def build_prompt(prompt_template, **kwargs):
    """将 Prompt 模板赋值"""
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)

# TODO How to split the system prompt and user prompt?
prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。

已知信息:
{context} # 检索出来的原始文档

用户问：
{query} # 用户的提问

如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
请不要输出已知信息中不包含的信息或答案。
请用中文回答用户问题。
"""

# paragraphs = extract_paragraphs_from_pdf("llama2.pdf", min_line_length=10)
# context = paragraphs[:4] # 选择前 4 个段落作为已知信息
# query = "llama2有多少参数?"
# print(get_completion(build_prompt(prompt_template, context=context, query=query), model="gpt-4o-mini"))

"""
Step 3: Embedding the data, Ready for Similarity Calculation
"""
def get_embeddings(texts, model="text-embedding-ada-002", dimensions=None):
    """获取文本的嵌入"""
    if model == "text-embedding-ada-002":
        dimensions = None

    if dimensions:
        resp = client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions
        )
    else:
        resp = client.embeddings.create(
            input=texts,
            model=model
        )
    return [data.embedding for data in resp.data]

def cosine_similarity(a, b):
    """计算两个向量的余弦相似度 - 越大越相似"""
    return dot(a, b) / (norm(a) * norm(b))
def l2_similarity(a, b):
    """计算两个向量的 L2 距离 - 越小越相似"""
    return norm(np.asarray(a) - np.asarray(b))

def compare_embeddings_similarity(query, texts, model="text-embedding-3-small", dimensions=1000, method="cosine"):
    """计算问题与文本的相似度"""
    documents_embeddings = get_embeddings(texts, model, dimensions)
    query_embedding = get_embeddings([query], model, dimensions)[0]
    if method == "cosine":
        return [cosine_similarity(query_embedding, emb) for emb in documents_embeddings]
    elif method == "l2":
        return [l2_similarity(query_embedding, emb) for emb in documents_embeddings]

# query = "国际争端"
# query = "global conflicts" # 且能支持跨语言检索
# documents = [
#     "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
#     "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
#     "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
#     "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
#     "我国首次在空间站开展舱外辐射生物学暴露实验",
# ]
# print(compare_embeddings_similarity(query, documents, model="text-embedding-ada-002", method="l2"))
