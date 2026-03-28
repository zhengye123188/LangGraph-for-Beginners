from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough  # 关键！

# 1. 模型
model = ChatOllama(
    model="qwen2:0.5b",
    temperature=0.2
)

# 2. 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "根据参考资料简洁专业回答：{context}"),
    ("user", "问题：{input}")
])

# 3. 向量存储
vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="nomic-embed-text"))

# 4. 添加资料
vector_store.add_texts([
    "减肥就是要少吃多练",
    "减肥期间饮食要清淡少油",
    "要多运动燃烧卡路里"
])

# ===================== 关键步骤 =====================
# 5. 创建检索器（自动从向量库找资料）
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 6. 构建自动 RAG 链（自动检索 + 自动填资料）
rag_chain = {
    "context": retriever,    # 自动检索！
    "input": RunnablePassthrough()
} | prompt | model | StrOutputParser()
# ====================================================

# 7. 直接提问！完全不用手动检索！
response = rag_chain.invoke("怎么减肥")
print(response)