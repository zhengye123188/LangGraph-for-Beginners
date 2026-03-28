from typing import TypedDict
from langgraph.graph import StateGraph,START,END

# ① 定义 State（共享便签纸）
class MyState(TypedDict):
    message: str # 便签纸上有一条消息

# ② 定义 Node（工人）
def greet(state:MyState):
    return {"message":"你好！"+state["message"]}

def respond(state:MyState):
    return {"message":state["message"]+"很高兴认识你！"}
# 用edge连接起来
graph = StateGraph(MyState)
graph.add_node("greet",greet)
graph.add_node("respond",respond)
graph.add_edge(START,"greet")
graph.add_edge("greet","respond")
graph.add_edge("respond",END)

# 编译并运行
app = graph.compile()
result = app.invoke({"message": "我是小明"})
print(result["message"])
