# ====================== LangChain 各类文档加载器实战（项目级数据加载场景） ======================
# 从langchain社区导入多种文档加载器，适配项目中不同格式文件/网页的数据读取需求
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

os.environ["USER_AGENT"] = "my-app"  # 随便写个名字就行
# ---------------------- 实战1：纯文本文件加载（项目中最基础的文本数据读取） ----------------------
# 初始化TextLoader：指定待加载的txt文件路径，设置编码格式为utf-8（避免中文乱码）
# file_path = "D:\\Developers\\Pycharm\\project\\agent-development-learning\\data\\test.txt"
# loader = TextLoader(file_path, encoding="utf-8")
# # 执行加载：返回Document对象列表，每个Document包含核心内容（page_content）和元数据（metadata）
# document = loader.load()
# 递归字符文本分割器，用于按自然段落分割大文档
# 1. 段落分隔符（\n\n）
# 2. 换行符（\n）
# 3. 空格（ ）
# 4. 字符（单个字）
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=50,#分段最大字符数
#     chunk_overlap=10, #分段之间允许重复的字符数
#     length_function=len
# )
# split_docs = splitter.split_documents(document)
# print(split_docs)
# ---------------------- 实战2：CSV文件加载（项目中表格数据读取与数据分析场景） ----------------------
# 初始化CSVLoader：指定csv文件路径，配置编码格式，自定义字段名
# loader = CSVLoader(
#     file_path="D:\\Developers\\Pycharm\\project\\agent-development-learning\\data\\stu.csv",
#     encoding ="UTF-8"
# )
# documents = loader.load()
# print(documents)
# print("------------------------------------------------")
# documents = loader.lazy_load()
# for document in documents:
#     print(document)

# ---------------------- 实战3：JSON文件加载（项目中结构化JSON数据提取场景） ----------------------
# 初始化JSONLoader：指定json文件路径，配置数据提取规则
# jq_schema=".articles[]"：使用jq语法，提取JSON中articles数组的每一个元素
# content_key="content"：指定将每个数组元素的"content"字段作为Document的核心内容
# loader = JSONLoader("data/test.json",jq_schema=".articles[]",content_key="content")
# print(f"json loader:{loader}")  # 打印JSONLoader实例信息
# docs = loader.load()  # 执行加载，提取符合规则的结构化数据
# print(docs)  # 打印加载后的Document列表（articles数组元素数量对应Document数量）
# print(len(docs))  # 打印提取的文档数量（等于articles数组的长度）
# loader = JSONLoader(
#     file_path="D:\\Developers\\Pycharm\\project\\agent-development-learning\\data\\test1.json",
#     jq_schema=".[].name",
#     text_content=False
# )
# documents = loader.load()
# print(documents)

# pdf加载器
loader = PyPDFLoader (
    file_path="D:\\Developers\\Pycharm\\project\\agent-development-learning\data\\test.pdf",
    mode = "single",# 只返回一个document对象
)
for doc in loader.lazy_load():
    print(doc)
