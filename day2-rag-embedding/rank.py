from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import chromadb
from chromadb.config import Settings
from nltk.tokenize import sent_tokenize
import json
from sentence_transformers import CrossEncoder

from rag_process_text import chunking_text_by_paragraphs
from rag_embedding import get_embeddings

paragraphs = chunking_text_by_paragraphs("llama2.pdf", page_numbers=[2, 3])
chroma_client = chromadb.Client(Settings(allow_reset=True))

# 为了演示，实际不需要每次 reset()
chroma_client.reset()

# 创建一个 collection
collection = chroma_client.get_or_create_collection(name="demo")

# top_k aNN nearest k neighbors
# top_n aNN nearest n percent neighbors
def semantic_search(query, top_k):
    '''检索向量数据库'''
    results = collection.query(
        query_embeddings=[get_embeddings(query)],
        n_results=top_k
    )
    return results


def split_text(paragraphs,chunk_size=300,overlap_size=100,min_line_length=10):
    '''按指定 chunk_size 和 overlap_size 交叠割文本'''
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i= 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev_len = 0
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap+chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks


chunks = split_text(paragraphs,300,100)

# 向 collection 中添加chunk与向量
collection.add(
    embeddings=[get_embeddings(chunk) for chunk in chunks],
    documents=chunks,
    metadatas=[{"source": "llama2.pdf"} for _ in chunks],
    ids=[f"ck_{i}" for i in range(len(chunks))]
)

user_query="how safe is llama 2"

search_results = semantic_search(user_query,5)

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
scores = model.predict([(user_query, doc) for doc in search_results['documents'][0]])
# 按得分排序
sorted_list = sorted(zip(scores,search_results['documents'][0]), key=lambda x: x[0], reverse=True)
for score, doc in sorted_list:
    print(f"{score}\t{doc}\n")

