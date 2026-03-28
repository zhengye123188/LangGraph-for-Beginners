from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 模拟的商品数据库
PRODUCTS = {
    "手机": {"price": 4999, "stock": 5},
    "耳机": {"price": 299, "stock": 0},
    "键盘": {"price": 599, "stock": 10},
}

class OrderProcess(TypedDict):
    item: str
    quantity: int
    price: int
    total_price: int
    stock: int
    result: str

def receive_order(state: OrderProcess):
    """接收订单，查出单价和库存"""
    product = PRODUCTS[state["item"]]  # ← 修正取值方式
    return {"price": product["price"], "stock": product["stock"]}

def calculate_order(state: OrderProcess):
    """计算总价 = 单价 × 数量"""
    total_price = state["price"] * state["quantity"]
    return {"total_price": total_price}

def check_stock(state: OrderProcess):
    """根据库存决定走哪条路"""
    if state["quantity"] <= state["stock"]:  # ← 修正逻辑方向
        return "confirm"
    else:
        return "reject"

def confirm(state: OrderProcess):
    """库存充足，确认订单"""
    return {"result": f"订单确认！{state['item']} x{state['quantity']}，总价：{state['total_price']}元"}

def reject(state: OrderProcess):
    """库存不足，拒绝订单"""
    return {"result": f"抱歉，{state['item']}暂时缺货"}

# ========== 组装图==========

graph = StateGraph(OrderProcess)

# 添加节点
graph.add_node("receive_order", receive_order)
graph.add_node("calculate_order", calculate_order)
graph.add_node("confirm", confirm)
graph.add_node("reject", reject)

# 添加直线边
graph.add_edge(START, "receive_order")
graph.add_edge("receive_order", "calculate_order")

# 添加条件边（岔路口）         判断函数返回的字符串  →  实际要去的节点名
graph.add_conditional_edges(
    "calculate_order", # 起点
    check_stock,{
    "confirm" : "confirm",
    "reject" : "reject"
})
# 两条路都通向终点
graph.add_edge("confirm", END)
graph.add_edge("reject", END)

# 编译
app = graph.compile()
# 测试1：买手机（有货）
r1 = app.invoke({"item": "手机", "quantity": 2,
                  "price": 0, "total_price": 0, "stock": 0, "result": ""})
print(r1["result"])

