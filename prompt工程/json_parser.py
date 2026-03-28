# ====================== 核心功能：使用 JsonOutputParser 解析器，将大模型返回结果转为标准 JSON 格式并可直接按键取值 ======================
# 从 langchain_core.prompts 模块导入 PromptTemplate 类，用于构建基础提示词模板
from langchain_core.prompts import PromptTemplate
# 从 langchain_openai 模块导入 ChatOpenAI 类，用于调用兼容 OpenAI 接口的大模型（此处对接阿里云通义千问）
from langchain_openai import ChatOpenAI
# 从 pydantic 模块导入 SecretStr 类，用于安全封装 API 密钥，避免明文泄露敏感信息
from pydantic import SecretStr
# 从 langchain_core.output_parsers 模块导入多种解析器
# StrOutputParser：纯字符串解析器，CommaSeparatedListOutputParser：逗号分隔列表解析器，JsonOutputParser：JSON格式解析器（核心使用）
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableLambda
# 注释：原始备用提示词模板（手动指定JSON结构，此处保留原始代码结构，未实际使用）
# prompt = PromptTemplate.from_template("返回JSON:{{'name':'姓名','age':'年龄'}},输入:{input}")
# 说明：模板中使用 {{ }} 是转义写法，最终会渲染为单个 { }，用于指定JSON格式中的大括号
def to_uppercase(text):
    return text.upper()  # Python内置功能：转大写

# 1. 实例化 ChatOpenAI 大模型对象，配置连接参数与生成属性
model = ChatOpenAI(
    model="qwen2:0.5b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("none"),
    temperature=0.1
)  # 设置模型生成温度（取值范围 0-2），0.7 兼顾回答的准确性和灵活性（值越低回答越严谨，更易生成标准JSON）

# 2. 实例化 JsonOutputParser JSON 解析器
# 核心本质：1. 自动识别大模型返回的 JSON 格式字符串，将其转换为 Python 可直接操作的字典（dict）对象
#          2. 若大模型返回非标准 JSON，会尝试容错解析；若完全不符合 JSON 格式，会抛出解析异常
parse = JsonOutputParser()

# 注释：原始备用链式调用（此处保留原始代码结构，未实际使用）
# chain = prompt | model | parse
# print(chain.invoke({"input":"我叫小王，今年18岁"}))

# 3. 构建提示词模板（核心：明确要求大模型返回指定结构的 JSON 格式，确保 JsonOutputParser 能正常解析）
# 模板中通过清晰的 JSON 示例结构，约束大模型返回包含 "answer"（答案文本）和 "confidence"（0-1区间置信度）的 JSON
# 问题:{question} 为动态变量，用于接收用户的具体问题
prompt = PromptTemplate.from_template("""
    回答以下问题，严格返回json格式，不需要其他任何文字:
    {{
        "answer":"答案文本",
        "confidence": 置信度(0-1)
    }}
    问题:{question}
""")
second_prompt = PromptTemplate.from_template("请用英文解释这个答案{answer}的理由")
parser = JsonOutputParser()
chain = prompt | model | parser | second_prompt | model | StrOutputParser() | RunnableLambda(to_uppercase)
# json不支持流式输出
print(chain.invoke({"question":"光年是时间单位还是长度单位？"}))