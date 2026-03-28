from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchainTools.weather_tool import get_weather

# 1. 创建模型（本地Ollama）
model = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.3,
)

# 2. 工具列表
tools = [get_weather]

# 3. ReAct提示词（手动定义，不依赖hub）
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

# 4. 创建智能体
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

# 5. 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

# 6. 调用
res = agent_executor.invoke({"input": "今天北京天气如何"})
print(res["output"])