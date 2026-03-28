from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
# ✅ 新版 Chroma（无警告）
from langchain_chroma import Chroma

# 1. Ollama 向量模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 外部向量存储（持久化，保存到本地 chroma_db 文件夹）
vector_store = Chroma(
    collection_name="nomic_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 数据永久存在这
)

# 3. 读取文本
loader = TextLoader(
    file_path="D:\\Developers\\Pycharm\\project\\agent-development-learning\\data\\test.txt",
    encoding="utf-8"
)
documents = loader.load()

# 4. 添加文档（和你之前写法一模一样）
vector_store.add_documents(
    documents=documents,
    ids=["id"+str(i) for i in range(1, len(documents)+1)]
)
# 下次运行直接加载，不用再读文件、加文档
# vector_store = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings
# )
# 5. 测试搜索（可选）
query = "我是谁"
docs = vector_store.similarity_search(query)
print("🔍 搜索结果：", docs[0].page_content)