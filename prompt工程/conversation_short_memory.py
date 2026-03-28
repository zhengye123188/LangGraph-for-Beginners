from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from pydantic import SecretStr

# 1. 模型
model = ChatOpenAI(
    model="qwen2:0.5b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("none"),
    temperature=0.3
)

# 2. 提示词（必须留 history 位置）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有记忆的聊天助手"),
    MessagesPlaceholder(variable_name="history"),  # 👈 历史消息位置
    ("human", "{input}")
])

# 3. 基础 chain
chain = prompt | model | StrOutputParser()

# ==========================
# 🌟 核心：Runnable 记忆包装器
# ==========================
store = {}  # 记忆存储（多用户会话）

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 用 RunnableWithMessageHistory 实现记忆
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,  # 获取历史
    input_messages_key="input",              # 输入字段
    history_messages_key="history"           # 历史字段（必须和prompt一致）
)

# ==========================
# 测试对话（带记忆）
# ==========================
if __name__ == "__main__":
    # 同一个 session_id 共享记忆
    config = {"configurable": {"session_id": "user_123"}}

    print(chain_with_memory.invoke({"input": "我叫张三"}, config=config))
    print(chain_with_memory.invoke({"input": "1+1等于几"}, config=config))
    print(chain_with_memory.invoke({"input": "我叫什么名字,不是你叫什么名字"}, config=config))  # 能记住！