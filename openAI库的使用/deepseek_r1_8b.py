# 导入LLM版Ollama（不是ChatOllama）
from langchain_community.llms import Ollama

# 初始化大语言模型（LLM）
llm = Ollama(
    model="deepseek-r1:8b",  # 你的本地模型名
    temperature=0.7,
    base_url="http://localhost:11434"
)

# 初始化上下文（LLM用纯字符串拼接，而非消息对象列表）
chat_history = ""
# 系统提示词（LLM需手动拼接到上下文里）
system_prompt = "你是一个AI智能助手，回答简洁易懂。\n"

# 交互提示
print("=== 大语言模型（LLM）流式多轮对话 ===")
print("输入 '退出'/'q'/'quit' 结束对话\n")

# 多轮对话主循环
while True:
    # 获取用户输入
    user_input = input("你：").strip()
    if not user_input:
        print("⚠️ 输入不能为空，请重新输入！")
        continue

    # 退出逻辑
    if user_input.lower() in ["退出", "q", "quit"]:
        print("模型：再见！")
        break

    # 拼接上下文（LLM核心：用字符串维护对话历史）
    prompt = system_prompt + chat_history + f"用户：{user_input}\n模型："

    # 流式调用LLM（核心区别：LLM的stream返回字符串片段）
    print("模型：", end="", flush=True)
    full_response = ""
    # LLM的stream直接返回字符串chunk，无需.content取值
    for chunk in llm.stream(prompt):
        if chunk.strip():  # 过滤空片段
            print(chunk, end="", flush=True)
            full_response += chunk
    print()  # 回复结束后换行

    # 更新上下文（纯字符串拼接，无AIMessage）
    chat_history += f"用户：{user_input}\n模型：{full_response}\n"