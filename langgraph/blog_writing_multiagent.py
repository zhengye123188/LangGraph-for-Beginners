from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
# 定义状态节点
class BlogState(TypedDict):
    topic: str              # 用户给的主题
    research: str           # 调研员收集的素材
    draft: str              # 写手写的文章
    feedback: str           # 编辑的审核意见
    revision_count: int     # 已修改次数（防止无限循环）
    final_article: str      # 最终通过的文章

model = ChatOpenAI(
    model = "deepseek-chat",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)
# Agent 1：调研员
def researcher(state: BlogState):
    prompt = f"""你是一个资深调研员。请围绕以下主题，收集3-5个关键要点，
    每个要点附带简短说明。只输出要点，不要写成文章。
    主题：{state["topic"]}"""
    response = model.invoke([HumanMessage(content=prompt)])
    return {"research":response.content}

# Agent 2:写手
def writer(state: BlogState):
    # 存储编辑的审核意见
    feedback_section=""
    if state.get("feedback"):
        feedback_section = f"\n\n编辑反馈（请据此修改）：{state['feedback']}\n之前的稿子"
    prompt = f"""你是一个专业博客写手。请根据以下素材撰写一篇简短的博客文章（200-300字）。
    要求：标题吸引人，结构清晰，语言生动。
    调研素材：
    {state["research"]}
    {feedback_section}"""
    response = model.invoke([HumanMessage(content=prompt)])
    print(f"写手第{state["revision_count"]}次：{response.content}")
    return {"draft":response.content,"revision_count":state.get("revision_count",0)+1}
# Agent3:编辑
def editor(state: BlogState):
    prompt = f"""你是一个严格的编辑。请审核以下博客文章，给出你的判断。
    如果文章质量合格（结构完整、语言流畅、观点清晰），请回复：通过
    如果需要修改，请回复：修改，然后给出具体的修改意见。
    文章：
    {state["draft"]}"""
    response = model.invoke([HumanMessage(content=prompt)])
    content = response.content
    print(f"编辑第{state["revision_count"]}次修改意见{content}")
    if "通过" in content and "修改" not in content:
        return {"feedback": "通过", "final_article": state["draft"]}
    else:
        return {"feedback": content}
# 编辑的路由判断
def should_revise(state: BlogState):
    # 超过3次修改，强制通过
    if state.get("revision_count", 0) >= 3:
        return "end"
    if state.get("feedback", "") == "通过":
        return "end"
    return "revise"

# 组装图
graph = StateGraph(BlogState)

graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("editor", editor)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "editor")

graph.add_conditional_edges(
    "editor",
    should_revise,
    {
        "revise": "writer",    # 打回给写手修改
        "end": END,            # 通过，结束
    }
)

app = graph.compile()

result = app.invoke({
    "topic": "为什么程序员应该学习 AI",
    "research": "",
    "draft": "",
    "feedback": "",
    "revision_count": 0,
    "final_article": "",
})

print("=" * 50)
print(f"经过 {result['revision_count']} 轮修改")
print("=" * 50)
print(result["final_article"])