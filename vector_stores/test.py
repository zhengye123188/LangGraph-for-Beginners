from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量相关
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(
    model = "qwen2:0.5b",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","以我提供的参考资料为主，简介且专业地回答用户问题，参考资料:{context}"),
        ("user","用户提问：{input}")
    ]
)

vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model="nomic-embed-text"))

vector_store.add_texts(["减肥就是要少吃多练","在减肥期间饮食很重要，要清淡少油减少热量摄入","要多运动增加卡路里消耗，多进行燃脂运动"])

input = "怎么减肥"

result = vector_store.similarity_search(input,2)

# 拼接向量库
context_all = ""
for item in result:
    context_all += item.page_content + "，"  # 加 .page_content 就不报错了

chain = prompt | model | StrOutputParser()

res = chain.invoke({"context":context_all,"input":input})
print(res)