from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import os

@tool
def pay_order(item: str,quantity: int) -> str:
    """用户下单购买商品.这是一个重要操作，会产生实际费用，需要询问用户是否继续"""
    prices = {"手机": 4999, "耳机": 299, "键盘": 599}
    price = prices.get(item, 0)
    total = price * quantity
    return f"订单已提交：{item} x{quantity}，总价{total}元"
@tool
def get_price(item: str) -> str:
    """查询商品价格。输入商品名称，返回价格信息。"""
    products = {
        "手机": 4999,
        "耳机": 299,
        "键盘": 599,
    }
    price = products.get(item)
    if price:
        return f"{item}的价格是{price}元"
    return f"没有找到{item}的价格信息"

@tool
def get_stock(item: str) -> str:
    """查询商品库存。输入商品名称，返回库存信息。"""
    stock = {
        "手机": 5,
        "耳机": 0,
        "键盘": 10,
    }
    qty = stock.get(item)
    if qty is not None:
        return f"{item}库存{qty}件" if qty > 0 else f"{item}暂时缺货"
    return f"没有找到{item}的库存信息"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。输入如 '4999 * 3'，返回计算结果。"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except:
        return "计算出错，请检查表达式"



llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

# 把工具列表告诉大模型
tools = [get_price, get_stock, calculate,pay_order]
# 工具节点（自动执行大模型请求的工具）
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)
# Agent 节点
def agent(state: MessagesState):
    system = SystemMessage(content="你是智能客服。可以查价格、查库存、做计算。")
    all_messages = [system] + state["messages"]
    response = llm_with_tools.invoke(all_messages)
    return {"messages": [response]}

# 组装图
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")

# 关键：用 tools_condition 自动判断
graph.add_conditional_edges(
    "agent",
    tools_condition,       # LangGraph 内置的判断函数
)
graph.add_edge("tools", "agent")   # 工具执行完，回到 agent 继续思考
# 改动1：创建 checkpointer
memory = MemorySaver()

# 改动2：compile 时传入 checkpointer
app = graph.compile(checkpointer=memory,interrupt_before=["tools"])

config = {"configurable": {"thread_id": "chat_001"}}

print("智能客服已启动！输入 exit 退出。")
print("-" * 40)

while True:
    user_input = input("用户：")
    if user_input == "exit":
        print("已退出对话")
        break
    result=app.invoke({
        "messages": [HumanMessage(content=user_input)]},
        config=config)
    # 检查是否暂停了（图还没走到 END）
    state = app.get_state(config)

    while state.next:  # state.next 不为空 = 图暂停了
        # 显示 AI 打算做什么
        last_msg = state.values["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"\n  AI 想调用: {tc['name']}")
                print(f"  参数: {tc['args']}")

        # 让用户决定
        confirm = input("\n确认执行？(y/n): ")

        if confirm.lower() == "y":
            # 继续执行
            result = app.invoke(None, config=config)
            state = app.get_state(config)
        else:
            # 拒绝：添加一条消息告诉 AI 用户取消了
            tool_call_id = last_msg.tool_calls[0]["id"]
            result = app.invoke(
                {"messages": [ToolMessage(
                    content="用户取消了这个操作",
                    tool_call_id=tool_call_id
                )]},
                config=config
            )
            state = app.get_state(config)

    print(f"客服: {result['messages'][-1].content}")
    print("-" * 40)