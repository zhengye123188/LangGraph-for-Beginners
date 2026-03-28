# 文本嵌入模型：将字符串作为输入，返回一个浮点数的列表（向量）
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
# 初始化嵌入对象模型,指定本地嵌入模型
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    show_progress = False
)
# 单条文本生成嵌入向量（用于查询场景，如用户问题）
query_text = "我是郑晔，赵承阳最严厉的父亲"
# 生成向量
# query_vector = embeddings.embed_query(query_text)
# print(f"向量维度{query_vector}")

# 批量生成
docs_texts= [
    "郑晔是赵承阳的严格之父",
    "我是郑晔，赵承阳最严厉的父亲",
    "赵承阳是郑晔最神的儿子",
    "赵承阳最严厉的父亲是郑晔"
]
docs_vectors = embeddings.embed_documents(docs_texts)
for i, vec in enumerate(docs_vectors):
    print(f"第{i+1}个向量（对应文本：{docs_texts[i]}）：")
    print(f"向量：{vec}")
