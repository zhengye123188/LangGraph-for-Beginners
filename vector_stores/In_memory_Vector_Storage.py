from idlelib.outwin import file_line_pats

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings  # 修复后
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import (
    TextLoader,        # 加载纯文本文件（.txt），项目中用于读取普通文本数据
    UnstructuredURLLoader,# 加载静态网页数据，项目中用于爬取无动态渲染的网页内容
    UnstructuredFileLoader,# 通用文件加载器，适配多种非结构化文件
    PyPDFLoader,       # 加载PDF文件，项目中用于读取PDF格式的文档/报告
    Docx2txtLoader,    # 加载Word文档（.docx），项目中用于读取办公文档数据
    CSVLoader,         # 加载CSV文件，项目中用于读取表格类数据（如销售数据、用户数据）
    UnstructuredHTMLLoader,# 加载HTML文件，项目中用于读取本地/远程HTML格式数据
    SeleniumURLLoader, # 加载动态渲染网页（JS渲染），项目中用于爬取需要动态加载的网页
    WebBaseLoader,     # 通用网页加载器，简化网页数据读取流程
    JSONLoader,        # 加载JSON文件，项目中用于读取结构化JSON数据并按需提取
)
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量相关
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
# 1. 加载 Ollama 本地 Embedding（无警告）
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 内存向量存储
file_path = "D:\\Developers\\Pycharm\\project\\agent-development-learning\\data\\test.txt"

loader = TextLoader(
    file_path,
    encoding="utf-8"
)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)

texts = text_splitter.split_documents(documents)

vector_store = InMemoryVectorStore.from_documents(
    texts,
    embedding=embeddings
)

# 3. 测试搜索
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
result = retriever.invoke("北邮和清北谁更胜一筹？")
print("搜索结果：", result[1].page_content)