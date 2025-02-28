"""
# !pip install chromadb
"""
from chromadb.config import Settings
from rag_embedding import get_embeddings
import chromadb
import json

"""
Step 4: Vector Store In Memory, Ready for storing and searching Embeddings
"""
class ChromaDbConnector:
    def __init__(self, collection_name, embedding_func, chroma_client=None):
        if chroma_client is None:
            chroma_client = chromadb.Client(Settings(allow_reset=True))
            # TODO Only for demo purpose, reset the collection
            chroma_client.reset()

        # Create a new collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_func = embedding_func

    '''添加文档到向量数据库'''
    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.embedding_func(documents),
            documents=documents,
            ids=[f"id{i}" for i in range(len(documents))]
        )

    '''检索向量数据库'''
    def search(self, query, top_n):
        return self.collection.query(
            query_embeddings=self.embedding_func([query]),
            n_results=top_n
        )


# # Run ChromaDB server
# # chroma run --path ./db_path
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)
# vector_db = ChromaDbConnector("llama2", get_embeddings, chroma_client)
vector_db = ChromaDbConnector("llama2", get_embeddings)

# paragraphs = extract_paragraphs_from_pdf(
#     "llama2.pdf",
#     page_numbers=[2, 3],
#     min_line_length=10
# )
# vector_db.add_documents(paragraphs)
# # Similarity search for the user query, with top 2 most similar paragraphs
# user_query = "Llama 2有多少参数" # "Does Llama 2 have a conversational variant?"
# results = vector_db.search(user_query, 2)
# print(f"Similarity: {results['distances']}")
# for para in results['documents'][0]:
#     print(para+"\n")

documents = [
    "玛丽患有肺癌，癌细胞已转移",
    "刘某肺癌I期",
    "张某经诊断为非小细胞肺癌III期",
    "小细胞肺癌是肺癌的一种"
]
# 关键字检索
query = "非小细胞肺癌的患者"
vector_db.add_documents(documents)
vector_search_results = vector_db.search(query, 3)

print(json.dumps(vector_search_results, indent=4, ensure_ascii=False))