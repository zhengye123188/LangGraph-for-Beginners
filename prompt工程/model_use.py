# 该代码功能：输入职位+学习方向，生成专业学习规划
# 本地大模型版本（Ollama），零费用、本地运行

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser

# ===================== 本地大模型配置（必须填） =====================
model = ChatOpenAI(
    model="qwen2:0.5b",       # 你本地 Ollama 运行的模型名
    base_url="http://localhost:11434/v1",  # 本地模型地址
    api_key=SecretStr("none"), # 本地模型随便填
    temperature=0.7
)

# ===================== 提示词模板 =====================
# 模板：职位 + 学习内容 → 生成学习规划
template = "我是一名{position}，帮我写一个{study}学习规划"

prompt_template = PromptTemplate(
    template=template,
    input_variables=["position", "study"]  # 修正拼写！
)

# ===================== 调用（正确传参） =====================
# 格式必须是字典！！！
prompt = prompt_template.invoke({
    "position": "大学生",
    "study": "Java"
})

# 调用模型
res = model.invoke(prompt)
print(res.content)
