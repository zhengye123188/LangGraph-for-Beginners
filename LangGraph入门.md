# LangGraph 从零到精通 —— 完整学习笔记

---

## 第 1 课：核心概念 —— State / Node / Edge

### 基础内容

LangGraph 是一个基于「图」结构来构建 AI 工作流的 Python 框架，属于 LangChain 生态，但可以独立使用。

### 核心概念（三件套）

| 概念 | 类比 | 作用 |
|------|------|------|
| **State（状态）** | 共享便签纸 | 所有节点共用的数据容器，每个节点都能读写 |
| **Node（节点）** | 工人 | 一个 Python 函数，负责完成一件具体的事 |
| **Edge（边）** | 走廊 | 决定做完 A 之后该去找谁 |

### 最小可运行代码

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# ① 定义 State（共享便签纸）
class MyState(TypedDict):
    message: str

# ② 定义 Node（工人）
def greet(state: MyState):
    return {"message": "你好！" + state["message"]}

def respond(state: MyState):
    return {"message": state["message"] + " 很高兴认识你！"}

# ③ 用 Edge 连起来
graph = StateGraph(MyState)
graph.add_node("greet", greet)
graph.add_node("respond", respond)
graph.add_edge(START, "greet")
graph.add_edge("greet", "respond")
graph.add_edge("respond", END)

app = graph.compile()
result = app.invoke({"message": "我是小明"})
print(result["message"])
# 输出: 你好！我是小明 很高兴认识你！
```

### 需要注意的点

- `TypedDict` 用来定义 State 的结构，规定便签纸上有哪些栏位
- 节点函数返回的字典会**合并**到 State，不是覆盖——只更新你返回的字段，其他字段不变
- `START` 和 `END` 是特殊标记，表示图的入口和出口
- `compile()` 把图编译成可运行的应用，`invoke()` 传入初始状态来运行

### State 设计方法

设计 State 的三步法：

1. 列出所有节点（工人）
2. 问每个节点：你需要读什么？你会产出什么？
3. 合并所有信息，去掉重复的

**核心原则：如果一个数据只在一个函数内部用到，就不需要放进 State；只有需要在节点之间传递的数据才放进去。**

---

## 第 2 课：条件路由 —— 让图学会做决定

### 核心内容

`add_conditional_edges` 让图能根据条件走不同路径，类似 if/else 但用在图的结构中。

### 完整代码：订单处理系统

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

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
    product = PRODUCTS[state["item"]]
    return {"price": product["price"], "stock": product["stock"]}

def calculate_order(state: OrderProcess):
    """计算总价"""
    total_price = state["price"] * state["quantity"]
    return {"total_price": total_price}

def check_stock(state: OrderProcess):
    """根据库存决定走哪条路"""
    if state["quantity"] <= state["stock"]:
        return "confirm"
    else:
        return "reject"

def confirm(state: OrderProcess):
    return {"result": f"订单确认！{state['item']} x{state['quantity']}，总价：{state['total_price']}元"}

def reject(state: OrderProcess):
    return {"result": f"抱歉，{state['item']}暂时缺货"}

graph = StateGraph(OrderProcess)
graph.add_node("receive_order", receive_order)
graph.add_node("calculate_order", calculate_order)
graph.add_node("confirm", confirm)
graph.add_node("reject", reject)

graph.add_edge(START, "receive_order")
graph.add_edge("receive_order", "calculate_order")

# 条件边：三个参数
graph.add_conditional_edges(
    "calculate_order",      # 参数1：从哪个节点出发
    check_stock,            # 参数2：判断函数（不加括号！）
    {                       # 参数3：判断结果 → 目标节点
        "confirm": "confirm",
        "reject": "reject",
    }
)

graph.add_edge("confirm", END)
graph.add_edge("reject", END)

app = graph.compile()
```

### add_conditional_edges 参数详解

```python
graph.add_conditional_edges(
    "calculate_order",   # 参数1：起点节点名（字符串）—— 在哪设岔路口
    check_stock,         # 参数2：判断函数（函数名，不加括号！）—— 谁来判断
    {                    # 参数3：路线映射字典 —— 每条路通向哪里
        "confirm": "confirm",
        "reject": "reject",
    }
)
```

- 参数3的左边是判断函数 return 的字符串，右边是 add_node 注册过的节点名
- 两边可以不同名，例如 `{"有货": "confirm", "没货": "reject"}`

### 遇到的困难与解决

**问题1：判断函数被当成节点注册**

```python
# 错误：check_stock 不是节点，不要 add_node
graph.add_node("check_stock", check_stock)  # ← 删掉这行

# 正确：check_stock 只作为判断函数传给 add_conditional_edges
graph.add_conditional_edges("calculate_order", check_stock, {...})
```

> 记住区分：节点（工人，用 add_node 注册）vs 判断函数（交警，传给 add_conditional_edges）

**问题2：判断函数名加了引号变成字符串**

```python
# 错误：加了引号变成字符串
graph.add_conditional_edges("calculate_order", "check_stock", {...})

# 正确：不加引号，传的是函数本身
graph.add_conditional_edges("calculate_order", check_stock, {...})
```

> 报错信息：`TypeError: Expected a Runnable, callable or dict. Instead got an unsupported type: <class 'str'>`

**问题3：字段名拼写不一致**

```python
# State 里定义的是 result，代码里写成了 resul 或 confrim
# Python 不会自动纠错，只有运行到才报 KeyError
```

> 解决：State 里的字段名和代码里使用的必须完全一致

**问题4：逻辑判断反了**

```python
# 错误：数量 > 库存时应该拒绝，不是确认
if state["quantity"] > state["stock"]:
    return "confirm"  # ← 反了

# 正确
if state["quantity"] <= state["stock"]:
    return "confirm"
```

**问题5：取值方式错误——逗号创建元组**

```python
# 错误：逗号创建了元组，不是字典
product = PRODUCTS[item]["price"], PRODUCTS[item]["stock"]

# 正确：先取整个字典
product = PRODUCTS[item]
return {"price": product["price"], "stock": product["stock"]}
```

---

## 第 3 课：接入大语言模型

### 核心内容

- 使用 `ChatOpenAI` 连接大模型（支持 OpenAI / DeepSeek / Ollama 等）
- 使用 `MessagesState` 管理对话消息列表
- `MessagesState` 的 messages 字段会**追加**而非覆盖

### 连接不同模型的方式

```python
from langchain_openai import ChatOpenAI

# DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="你的Key",
)

# Ollama 本地模型
llm = ChatOpenAI(
    model="qwen2.5:7b",                        # ollama list 查看可用模型
    base_url="http://localhost:11434/v1",        # Ollama 默认地址
    api_key="ollama",                            # 随便填，Ollama 不验证
)
```

### 完整聊天机器人代码

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

def chatbot(state: MessagesState):
    system = SystemMessage(content="你是一个友好的中文客服助手，回答要简洁。")
    all_messages = [system] + state["messages"]
    response = llm.invoke(all_messages)
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="你好，手机多少钱？")]
})
print(result["messages"][-1].content)
```

### chatbot 节点逐行解析

```python
def chatbot(state: MessagesState):
    # 1. 系统提示：给 AI 的角色设定，用户看不到
    system = SystemMessage(content="你是一个友好的中文客服助手")

    # 2. 拼接：系统提示排最前，用户消息跟在后面
    all_messages = [system] + state["messages"]

    # 3. 调用大模型，返回 AIMessage 对象
    response = llm.invoke(all_messages)

    # 4. 包成列表返回，MessagesState 会自动追加到消息列表末尾
    return {"messages": [response]}
```

### 三种消息类型

| 类型 | 用途 |
|------|------|
| `SystemMessage` | 给 AI 的角色设定，用户看不到 |
| `HumanMessage` | 用户说的话 |
| `AIMessage` | AI 的回复（由 `llm.invoke()` 返回） |

### 需要注意的点

- 这一课的代码只实现了**手动记忆**，每次需要手动带上历史消息
- `result["messages"][-1].content`：`[-1]` 取列表最后一个元素，`.content` 取文本内容
- `return {"messages": [response]}` 外面必须包列表 `[]`，否则 MessagesState 不知道怎么追加

### 遇到的困难与解决

**问题：用了 `openai.OpenAI` 而不是 `langchain_openai.ChatOpenAI`**

```python
# 错误：原生 OpenAI SDK，没有 bind_tools 等方法
from openai import OpenAI
llm = OpenAI(...)

# 正确：LangChain 封装版，有 bind_tools、invoke 等方法
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(...)
```

> 在 LangGraph 项目里始终用 `ChatOpenAI`

---

## 第 4 课：LLM 智能路由 —— 让大模型自己做决定

### 核心内容

第 2 课的条件路由靠手写 if/else，第 4 课让大模型分析用户意图来自动路由。

### State 设计对比

```python
# 第2课：多个细粒度字段
class OrderProcess(TypedDict):
    item: str
    quantity: int
    price: int        # 多个节点需要传递，必须放进 State
    stock: int        # 多个节点需要传递，必须放进 State
    result: str

# 第4课：统一出口字段
class ServiceState(TypedDict):
    user_input: str       # 入口数据
    intent: str           # 路由依据（AI 判断的意图）
    query_result: str     # 中间结果的统一出口
    response: str         # 最终回答
```

**为什么不用 price 和 stock？** 因为它们只在单个函数内部用完就变成文字了，没有其他节点需要原始数值。`query_result` 统一成一个字段，三条路径都往里写，`respond` 节点只需要读一个字段。

**query_result vs response 的区别：** query_result 是原始数据（"手机价格4999元"），response 是大模型润色后的话术（"您好！手机目前售价4999元，性价比很高哦"）。

### AI 路由器核心代码

```python
def router(state: ServiceState):
    prompt = f"""判断以下用户消息的意图，只回复一个词：
- 回复"价格"：如果用户在问商品价格
- 回复"库存"：如果用户在问有没有货
- 回复"闲聊"：如果是其他内容

用户消息：{state["user_input"]}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip()
    return {"intent": intent}
```

### 需要注意的点

- `router` 是节点（用 add_node），`route_by_intent` 是判断函数（传给 add_conditional_edges）
- invoke 时必须给所有 State 字段设初始值，否则报 KeyError

### 遇到的困难与解决

**问题1：invoke 时缺少初始值**

```python
# 错误：缺少 intent、query_result、response
app.invoke({"user_input": "手机多少钱？"})

# 正确：所有字段都要给初始值
app.invoke({"user_input": "手机多少钱？",
            "intent": "", "query_result": "", "response": ""})
```

**问题2：respond 函数里字段名写错**

```python
# 错误：State 里定义的是 query_result，写成了 result
state["result"]

# 正确
state["query_result"]
```

---

## 第 5 课：工具调用 —— 让智能体能做事

### 核心内容

- 用 `@tool` 装饰器定义工具
- 用 `bind_tools` 把工具告诉大模型
- 用 `ToolNode` + `tools_condition` 自动处理工具调用循环

### 工具定义

```python
from langchain_core.tools import tool

@tool
def get_price(item: str) -> str:
    """查询商品价格。输入商品名称，返回价格信息。"""
    products = {"手机": 4999, "耳机": 299, "键盘": 599}
    price = products.get(item)
    if price:
        return f"{item}的价格是{price}元"
    return f"没有找到{item}的价格信息"
```

**工具参数的运作方式：**

- 参数类型由你定义（`item: str`）
- 参数值由 AI 自动填充（用户说"手机多少钱"，AI 自动填 `item="手机"`）
- 三引号描述非常重要，AI 通过它判断什么时候该用哪个工具

### 完整代码

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [get_price, get_stock, calculate]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def agent(state: MessagesState):
    system = SystemMessage(content="你是智能客服。可以查价格、查库存、做计算。")
    all_messages = [system] + state["messages"]
    response = llm_with_tools.invoke(all_messages)
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")   # 工具执行完，回到 agent 继续思考

app = graph.compile()
```

### 工具调用的循环机制

```
agent 思考 → 需要工具 → tools 执行 → 结果回传给 agent → 继续思考
                                                    ↓
                                              不需要工具 → END
```

AI 可以连续调用多个工具，也可以看到结果后再调一个，直到它觉得信息够了。

### 需要注意的点

- `tools_condition` 是 LangGraph 内置的判断函数，自动检查 AI 回复里有没有工具调用请求
- `ToolNode` 自动执行工具并把结果放回消息列表
- 不是所有模型都支持工具调用，`deepseek-r1` 不支持，推荐用 `qwen2.5` 或 `deepseek-chat`

### 遇到的困难与解决

**问题：模型不支持工具调用**

```
openai.BadRequestError: deepseek-r1:8b does not support tools
```

> 解决：换成支持 tool calling 的模型，如 `qwen2.5:7b`、`llama3.1:8b`

---

## 第 6 课：记忆与持久化

### 核心内容

使用 `MemorySaver` + `thread_id` 实现自动对话记忆。

### 只需两行代码

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
```

### 使用方式

```python
config = {"configurable": {"thread_id": "user_A"}}

# 第一轮
app.invoke({"messages": [HumanMessage("手机多少钱？")]}, config=config)

# 第二轮：不用带历史！自动记住
app.invoke({"messages": [HumanMessage("买3个呢？")]}, config=config)
```

### MemorySaver 内部原理

```python
# 本质就是一个字典，用 thread_id 做 key
storage = {
    "user_A": {messages: [完整对话历史]},
    "user_B": {messages: [另一个对话历史]},
}
```

每次 invoke 时：
1. 从 config 取出 thread_id
2. 用 thread_id 去 storage 读档
3. 找到 → 把新消息追加到历史后面；没找到 → 新对话
4. 运行图
5. 运行完毕，把最终 State 存档

### thread_id 的作用

- 同一个 thread_id = 共享记忆的同一个对话
- 不同的 thread_id = 完全隔离的独立对话
- 固定写法：`{"configurable": {"thread_id": "你起的名字"}}`

### 多轮交互式对话测试

```python
config = {"configurable": {"thread_id": "chat_001"}}

print("智能客服已启动！输入 exit 退出。")
while True:
    user_input = input("你: ")
    if user_input.strip().lower() == "exit":
        print("再见！")
        break
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    print(f"客服: {result['messages'][-1].content}")
```

### 需要注意的点

- `MemorySaver` 存在内存里，程序关掉就没了
- 生产环境用 `SqliteSaver` 或 `PostgresSaver` 持久化到数据库
- 必须要有 checkpointer 才能使用 interrupt（第 7 课）

---

## 第 7 课：人机协作（Human-in-the-Loop）

### 核心内容

使用 `interrupt_before` 在关键操作前暂停，等人类确认后再继续。

### 设置暂停点

```python
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"]    # 执行工具前暂停
)
```

### 暂停与继续的机制

```python
# 1. invoke 可能在工具调用前暂停
result = app.invoke({"messages": [HumanMessage("帮我买2个手机")]}, config=config)

# 2. 检查是否暂停了
state = app.get_state(config)
# state.next 不为空 = 图暂停了

# 3a. 用户确认 → 传 None 继续执行
result = app.invoke(None, config=config)

# 3b. 用户拒绝 → 传 ToolMessage 告知取消
from langchain_core.messages import ToolMessage
tool_call_id = last_msg.tool_calls[0]["id"]
result = app.invoke(
    {"messages": [ToolMessage(content="用户取消了", tool_call_id=tool_call_id)]},
    config=config
)
```

### 确认 vs 拒绝的区别

| 操作 | 传入什么 | 发生什么 |
|------|----------|----------|
| 确认 | `None` | 图从暂停处继续，工具真正执行 |
| 拒绝 | `ToolMessage("用户取消了")` | 跳过工具执行，AI 收到"取消"消息后改口回复 |

### 需要注意的点

- `interrupt_before` 里的节点名必须是 `add_node` 注册过的
- 必须有 `checkpointer`，否则暂停后状态丢失无法继续
- `tool_call_id` 是必须的，用来配对工具调用请求和响应
- 拒绝时构造的 ToolMessage 是"假的工具结果"，AI 以为工具返回了取消信息

### 遇到的困难与解决

**问题：interrupt_before 写了不存在的节点名**

```python
# 错误：图里没有 "pay_order" 这个节点
app = graph.compile(checkpointer=memory, interrupt_before=["pay_order"])

# 正确：用 add_node 注册过的名字
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])
```

---

## 第 8 课：多智能体协作

### 核心内容

多个 Agent 在同一个图里协作，每个 Agent 是一个节点，有自己的系统提示和专长。

### 完整代码：博客写作团队

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="你的Key",
)

class BlogState(TypedDict):
    topic: str              # 用户给的主题
    research: str           # 调研员收集的素材
    draft: str              # 写手写的文章
    feedback: str           # 编辑的审核意见
    revision_count: int     # 已修改次数（防止无限循环）
    final_article: str      # 最终通过的文章

# Agent 1：调研员
def researcher(state: BlogState):
    prompt = f"""你是资深调研员。围绕以下主题收集3-5个关键要点。
主题：{state["topic"]}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"research": response.content}

# Agent 2：写手
def writer(state: BlogState):
    feedback_section = ""
    if state.get("feedback"):
        feedback_section = f"\n编辑反馈：{state['feedback']}\n之前的稿子：{state['draft']}"

    prompt = f"""你是专业博客写手。根据素材撰写200-300字博客文章。
调研素材：{state["research"]}
{feedback_section}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "draft": response.content,
        "revision_count": state.get("revision_count", 0) + 1
    }

# Agent 3：编辑
def editor(state: BlogState):
    prompt = f"""你是严格的编辑。审核文章质量。
合格回复"通过"，否则回复"修改"并给出意见。
文章：{state["draft"]}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if "通过" in content and "修改" not in content:
        return {"feedback": "通过", "final_article": state["draft"]}
    else:
        return {"feedback": content}

# 路由判断
def should_revise(state: BlogState):
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
graph.add_conditional_edges("editor", should_revise, {
    "revise": "writer",
    "end": END,
})

app = graph.compile()

result = app.invoke({
    "topic": "为什么程序员应该学习 AI",
    "research": "", "draft": "", "feedback": "",
    "revision_count": 0, "final_article": "",
})
print(f"经过 {result['revision_count']} 轮修改")
print(result["final_article"])
```

### 需要注意的点

- `revision_count` 是安全机制，防止编辑永远不满意导致无限循环
- `writer` 节点根据 State 里有没有 feedback 来决定是首次写作还是修改
- 编辑打回 → 写手修改 → 编辑再审——这个循环和第 5 课工具调用的循环是同一种模式
- 用 `state.get("key", 默认值)` 而不是 `state["key", 默认值]`

### 遇到的困难与解决

**问题1：HumanMessage 参数名写错**

```python
# 错误：参数名是 text
HumanMessage(text=prompt)

# 正确：参数名是 content
HumanMessage(content=prompt)
```

**问题2：state 取默认值写法错误**

```python
# 错误：方括号里放两个值，Python 以为是元组 key
state["revision_count", 0]

# 正确：用 .get() 方法
state.get("revision_count", 0)
```

---

## 全课程常见错误速查表

| 错误现象 | 原因 | 解决 |
|----------|------|------|
| `TypeError: Expected callable, got str` | 函数名加了引号 | 去掉引号：`check_stock` 不是 `"check_stock"` |
| `KeyError: 'xxx'` | State 字段名拼写不一致或缺少初始值 | 检查拼写；invoke 时给所有字段设初始值 |
| `Interrupt node 'xxx' not found` | interrupt_before 里的名字不存在 | 用 add_node 注册过的节点名 |
| `does not support tools` | 模型不支持工具调用 | 换模型：`qwen2.5:7b`、`deepseek-chat` |
| `'OpenAI' has no attribute 'bind_tools'` | 用了原生 OpenAI SDK | 改用 `from langchain_openai import ChatOpenAI` |
| `HumanMessage content is None` | 参数名写成了 `text` | 改成 `content=prompt` |
| `state["key", default]` 报错 | Python 语法错误 | 改成 `state.get("key", default)` |

---

## 知识体系总结

```
第1课 State/Node/Edge      ── 基础三件套
第2课 条件路由               ── 图能做判断
第3课 接入大模型             ── 节点变智能
第4课 LLM智能路由           ── AI自己选路径
第5课 工具调用               ── 智能体能做事
第6课 记忆持久化             ── 自动记住对话
第7课 人机协作               ── 关键操作需确认
第8课 多智能体协作           ── 多Agent分工合作
```

所有高级功能都建立在前两课的基础上：State 传递数据，Node 执行逻辑，Edge（包括条件边）控制流程。掌握了这三个概念，后面的课程都是在这个框架上添砖加瓦。
