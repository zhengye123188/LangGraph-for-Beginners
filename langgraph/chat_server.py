from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
# 调用模型
model = ChatOpenAI(
    model = "deepseek-r1:8b",
    base_url="http://localhost:11434/v1",   # ← Ollama 的本地地址
    api_key="ollama",                       # ← 随便填，Ollama 不验证 key
    temperature=0.5
)
# 模拟商品数据库
PRODUCTS = {
    "手机": {"price": 4999, "stock": 5},
    "耳机": {"price": 299, "stock": 0},
    "键盘": {"price": 599, "stock": 10},
}
# state
class ServiceState(TypedDict):
    user_input: str # 用户输入
    intent: str # 判断用户意图
    result: str # 查询结果
    response: str # 大模型润色后的回答
# 意图感知
def router(state:ServiceState):
    """Router for priming AI behavior."""
    prompt = f"""判断以下用户消息的意图，只回复一个词
    - 回复"价格"：如果用户在问商品价格、多少钱
    - 回复"库存"：如果用户在问有没有货、还剩多少
    - 回复"闲聊"：如果是其他内容
    用户消息：{state["user_input"]}
    """
    response = model.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip()
    return {"intent": intent}

# 查询价格节点
def check_price(state:ServiceState):
    """Check the price of priming AI behavior."""
    for name, info in PRODUCTS.items():
        if name in state["user_input"]:
            return {"result": f"{name}的价格是{info['price']}元"}
        return {"result":"抱歉，当前仓库当中没有相关商品"}

# 节点2b：查库存
def check_stock(state: ServiceState):
    for name, info in PRODUCTS.items():
        if name in state["user_input"]:
            if info["stock"] > 0:
                return {"result": f"{name}有货，剩余{info['stock']}件"}
            else:
                return {"result": f"{name}暂时缺货"}
    return {"query_result": "抱歉，没有找到您询问的商品"}

# 节点2c：闲聊
def chat(state: ServiceState):
    response = model.invoke([
        SystemMessage(content="你是友好的客服，简短回答。"),
        HumanMessage(content=state["user_input"]),
    ])
    return {"query_result": response.content}

# 节点3：用大模型组织最终回复
def respond(state: ServiceState):
    prompt = f"""根据以下信息，用友好的语气回复客户。

客户问题：{state["user_input"]}
查询结果：{state["result"]}

请用一两句话自然地回答："""

    response = model.invoke([HumanMessage(content=prompt)])
    return {"response": response.content}

# 路由判断函数（不是节点！）
def route_by_intent(state: ServiceState):
    intent = state["intent"]
    if "价格" in intent:
        return "check_price"
    elif "库存" in intent:
        return "check_stock"
    else:
        return "chat"

# 组装
graph = StateGraph(ServiceState)

graph.add_node("router", router)
graph.add_node("check_price", check_price)
graph.add_node("check_stock", check_stock)
graph.add_node("chat", chat)
graph.add_node("respond", respond)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_by_intent,
    {
        "check_price": "check_price",
        "check_stock": "check_stock",
        "chat": "chat",
    }
)

# 三条路都汇聚到 respond
graph.add_edge("check_price", "respond")
graph.add_edge("check_stock", "respond")
graph.add_edge("chat", "respond")
graph.add_edge("respond", END)

app = graph.compile()

# 测试1：问价格
r1 = app.invoke({"user_input": "手机多少钱？",
                  "intent": "", "result": "", "response": ""})
print(r1["response"])

# 测试2：问库存
r2 = app.invoke({"user_input": "耳机还有货吗？",
                  "intent": "", "result": "", "response": ""})
print(r2["response"])

# 测试3：闲聊
r3 = app.invoke({"user_input": "今天天气真好啊",
                  "intent": "", "result": "", "response": ""})
print(r3["response"])