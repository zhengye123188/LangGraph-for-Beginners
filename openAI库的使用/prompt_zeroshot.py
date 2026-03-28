from langchain_core.prompts import PromptTemplate
# 核心修改：导入新版OllamaLLM（替代旧版Ollama）
from langchain_ollama import OllamaLLM

# 初始化新版LLM模型（参数完全不变）
llm = OllamaLLM(
    model="deepseek-r1:8b",
    temperature=0.8  # 可选：起名场景调高随机性
)

# 提示词模板逻辑完全不变
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{last_name}，刚生了一个{gender}，请帮忙起5个好听的名字，每个名字附1句寓意解释，语言简洁。"
)

# # 动态填充参数
# prompt_text = prompt_template.format(last_name="郑", gender="女儿")
# 链式调用：直接传入参数字典，无需手动format
chain = prompt_template | llm
result = chain.invoke({"last_name": "郑", "gender": "女儿"})
print(result)