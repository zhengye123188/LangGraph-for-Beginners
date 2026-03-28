# ====================== 第一部分：LangChain 聊天提示词模板实战 ======================
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

# ✅ 正确导入（新版 langchain_openai，无警告）
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 方式1：直接通过ChatPromptTemplate.from_messages快速创建聊天提示词模板
chat_template = ChatPromptTemplate.from_messages([
    ("system","你是一个助手AI，名字是{name}"),
    ("human","你好，最近怎么样?"),
    ("ai","我很好谢谢"),
    ("human","{user_input}"),
])

message = chat_template.format_messages(name="HideOnBoss",user_input="你最喜欢的编程语言是什么?")
print(message)

# ====================== 方式2：先构建单条消息模板，再组合成聊天模板 ======================
system_template = SystemMessagePromptTemplate.from_template(
    "你是一个{role}，请用{language}回答"
)
user_template = HumanMessagePromptTemplate.from_template(
    "{question}"
)
chat_template = ChatPromptTemplate.from_messages([
    system_template,
    user_template,
])

message = chat_template.format_messages(
    role="学生",
    language="英文",
    question="你最喜欢什么学科?"
)
print(message)

# ====================== ✅ 修复：正确调用本地模型 ======================
model = ChatOpenAI(
    model="qwen2:0.5b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("dummy_key"),  # 必须写！本地模型随便填
    temperature=0.7,
)

# 调用模型
response = model.invoke(message)

# 输出结果
print("\n===== 模型回答 =====")
print(response.content)