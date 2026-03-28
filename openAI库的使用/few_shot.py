from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama import OllamaLLM

# 1.定义示例模板
single_example_template = """
示例{num}:
词汇{word}:
反义词{antonym}:
"""
# 2.初始化单个示例模板
example_template = PromptTemplate.from_template(single_example_template)
# 3.给fewshot
examples_data = [
    {"num":1, "word": "开心", "antonym": "难过"},
    {"num":2, "word": "快速", "antonym": "缓慢"},
    {"num":3, "word": "光明", "antonym": "黑暗"},
]
# 3. 用FewShotPromptTemplate一键组装少样本模板（核心简化点）
few_shot_template = FewShotPromptTemplate(
    examples=examples_data,                # 示例数据
    example_prompt=example_template,    # 单个示例的模板
    prefix="根据输入词汇输出对应的反义词，仅返回反义词，无需额外解释",  # 任务指令（示例前的说明）
    suffix="词汇：{target_word}\n反义词：",  # 待执行任务（示例后的占位符）
    input_variables=["target_word"]   # 最终需要传入的动态参数
)
# 4. 初始化模型+链式调用（极简调用）
llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.1)
chain = few_shot_template | llm

# 5. 测试调用（一行搞定）
print(chain.invoke({"target_word": "温暖"}).strip())  # 输出：寒冷

