# ====================== 核心功能：使用 LCEL 管道串联提示词与模型，生成产品卖点文案 ======================
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 如果你本地没有 main.py 这行直接删掉！！！
# from main import responses

# 1. 创建提示词模板
prompt_template = PromptTemplate(
    input_variables=["name"],
    template="""
    你是一个文案高手，专门为{name}设计文案，列举三个卖点
    """,
)

# 2. 本地模型配置（Ollama qwen2:0.5b）
model = ChatOpenAI(
    model="qwen2:0.5b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("none"),
    temperature=0.7
)

# ====================== ✅ 新版写法：用 | 管道替代 LLMChain（官方推荐） ======================
parser = StrOutputParser()
chain = prompt_template | model |parser

# 调用
response = chain.invoke({"name": "智能手机"})

# 输出结果
print(response)