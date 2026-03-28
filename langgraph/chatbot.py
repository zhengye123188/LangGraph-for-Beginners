import os

from langchain_ollama import ChatOllama
from openai import OpenAI
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState,StateGraph,START,END


# MessagesState 内部长这样：
# class MessagesState(TypedDict):
#     messages: list    # 自动追加，不会覆盖之前的消息
model = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.3,
)

# 用MessagesState管理会话
# 之前我们自己定义 TypedDict，但对于聊天场景
# LangGraph 提供了一个现成的 State 叫 MessagesState
# 它自动帮你管理消息列表
def chatbot(state:MessagesState):
    system =  SystemMessage(content="你是一个ai智能助手")
    all_messages = [system]+state["messages"]
    # 调用大模型
    response = model.invoke(all_messages)

    # 返回 AI 的回复（会自动追加到 messages 列表末尾）
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot",chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()
# 第一轮对话
result = app.invoke({
    "messages": [HumanMessage(content="你好，请问你们的手机多少钱？")]
})
print(result["messages"][-1].content)

# 第二轮对话（把上一轮的消息带上，实现"记忆"）
result2 = app.invoke({
    "messages": result["messages"] + [HumanMessage(content="有什么颜色可以选？")]
})
print(result2["messages"][-1].content)