from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 初始化ChatOllama模型（本地Ollama服务需提前启动）
model = ChatOllama(
    model="deepseek-r1:8b",  # 替换为你的本地模型名（如llama3、qwen:7b）
)

# 初始化对话上下文（系统提示词定义模型角色）
chat_history = [
    SystemMessage(content="你是一个友好的技术助手，回答简洁易懂，使用中文回复")
]

# 多轮对话主循环
while True:
    # 获取用户输入并处理空输入
    user_input = input("你：").strip()
    if not user_input:
        print("⚠️ 输入不能为空，请重新输入！")
        continue

    # 退出逻辑
    if user_input.lower() in ["退出", "q", "quit"]:
        print("模型：再见！")
        break

    # 将用户输入加入上下文
    chat_history.append(HumanMessage(content=user_input))

    # 流式调用模型并实时输出（核心逻辑）
    print("模型：", end="", flush=True)
    full_response = ""
    for chunk in model.stream(chat_history):
        # 过滤空片段，实时打印
        if chunk.content.strip():
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    print()  # 回复结束后换行

    # 将完整回复加入上下文（实现多轮记忆）
    chat_history.append(AIMessage(content=full_response))