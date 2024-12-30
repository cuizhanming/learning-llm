# !pip install elasticsearch7
# !pip install helpers

"""
ElasticSearch Server
> docker-compose -f docker-compose.yml up -d
"""

from elasticsearch7 import helpers
from elasticsearch7.client import Elasticsearch
from rag_process_text import to_keywords_chinese  # 使用中文的关键字提取函数
from dotenv import load_dotenv, find_dotenv
import time
import os
import json

_ = load_dotenv(find_dotenv(), verbose=True)  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

class MyEsConnector:
    def __init__(self, es_client, index_name, keyword_fn):
        self.es_client = es_client
        self.index_name = index_name
        self.keyword_fn = keyword_fn

    def add_documents(self, documents):
        """文档灌库"""
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
        self.es_client.indices.create(index=self.index_name)
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "keywords": self.keyword_fn(doc),
                    "text": doc,
                    "id": f"doc_{i}"
                }
            }
            for i, doc in enumerate(documents)
        ]
        helpers.bulk(self.es_client, actions)
        time.sleep(1)

    def search(self, query_string, top_n=3):
        """检索"""
        search_query = {
            "match": {
                "keywords": self.keyword_fn(query_string)
            }
        }
        res = self.es_client.search(
            index=self.index_name, query=search_query, size=top_n)
        return {
            hit["_source"]["id"]: {
                "text": hit["_source"]["text"],
                "rank": i,
            }
            for i, hit in enumerate(res["hits"]["hits"])
        }

# 引入配置文件
ELASTICSEARCH_BASE_URL = os.getenv('ELASTICSEARCH_BASE_URL')
ELASTICSEARCH_USERNAME= os.getenv('ELASTICSEARCH_USERNAME')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')
es = Elasticsearch(
    hosts=[ELASTICSEARCH_BASE_URL],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),  # 用户名，密码
)

# 创建 ES 连接器
es_connector = MyEsConnector(es, "demo_es_rrf", to_keywords_chinese)

# 文档灌库
documents = [
    "玛丽患有肺癌，癌细胞已转移",
    "刘某肺癌I期",
    "张某经诊断为非小细胞肺癌III期",
    "小细胞肺癌是肺癌的一种"
]
# 关键字检索
query = "非小细胞肺癌的患者"
es_connector.add_documents(documents)
keyword_search_results = es_connector.search(query, 3)
print(json.dumps(keyword_search_results, indent=4, ensure_ascii=False))