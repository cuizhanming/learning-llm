"""
# !pip install sentence_transformers
"""
import json

from sentence_transformers import CrossEncoder
from rag_retrieval_vector import vector_db

def rank_documents_with_llm(rank_model, query, documents, top_k=5):
    scores = rank_model.predict([(query, doc) for doc in documents])
    # Sort the documents by the scores
    sorted_results = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    for score, doc in sorted_results[:top_k]:
        print(f"Score: {score:.4f}\n{doc}\n")
    return sorted_results[:top_k]

# # How does this one use Hugging Face's Sentence Transformers model?
# model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512) # English model, smaller version
# # model = CrossEncoder('BAAI/bge-reranker-large', max_length=512) # Chinese model, larger version
# user_query = "how safe is llama 2"
# search_results = vector_db.search(user_query, 3)
# rank_documents_with_llm(model, user_query, search_results['documents'][0])

def rank_vector_documents(documents):
    return {
        "doc_"+str(documents.index(doc)): {
            "text": doc,
            "rank": i
        }
        for i, doc in enumerate(documents)
    }

# reformatted = rank_vector_documents(search_results['documents'][0])
# print(json.dumps(reformatted, indent=4, ensure_ascii=False))

def rrf_reciprocal_rank_fusion(ranks, k=1):
    ret = {}
    # 遍历每次的排序结果
    for rank in ranks:
        # 遍历排序中每个元素
        for i, val in rank.items():
            if i not in ret:
                ret[i] = {"score": 0, "text": val["text"]}
            # 计算 RRF 得分
            ret[i]["score"] += 1.0 / (k + val["rank"])
    # 按 RRF 得分排序，并返回
    return dict(sorted(ret.items(), key=lambda item: item[1]["score"], reverse=True))

from rag_retrieval_elastic import keyword_search_results
from rag_retrieval_vector import vector_search_results
reranked = rrf_reciprocal_rank_fusion([keyword_search_results, rank_vector_documents(vector_search_results['documents'][0])])
print(json.dumps(reranked, indent=4, ensure_ascii=False))
