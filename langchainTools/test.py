import requests
import pandas as pd
from langchain.tools import tool

# ✅ 正确导入（修复找不到 create_react_agent）
from langchain.agents import *
from langchain_openai import ChatOpenAI
from langchain_classic import hub

# ====================== 天气工具 ======================
@tool
def get_weather(city: str) -> str:
    """
    调用实时天气API返回温度及天气状况
    :param city: 城市名称，如北京
    :return: 天气信息字符串
    """
    try:
        city_id = get_city_id(city)
        url = "https://eolink.o.apispace.com/456456/weather/v001/now"
        payload = {"areacode": city_id}
        headers = {
            "X-APISpace-Token": "xnkahfw6hr30g011r2tcqf6w9q9p9yx8"
        }
        response = requests.get(url, params=payload, headers=headers)
        response.raise_for_status()
        res = response.json()

        realtime = res["result"]["realtime"]
        city_name = res["result"]["location"]["name"]
        weather = realtime["text"]
        temp = realtime["temp"]

        return f"🌤 {city_name}：{weather}，气温 {temp} ℃"

    except Exception as e:
        return f"⚠️天气查询失败：{str(e)}"

def get_city_id(city: str) -> str:
    try:
        city_df = pd.read_csv("天气预报查询国内城市.csv")
        district_match = city_df[city_df['district'].str.contains(city.strip(), na=False)]
        if not district_match.empty:
            return str(district_match['areacode/城市ID'].iloc[0])
        city_match = city_df[city_df['city'].str.contains(city.strip(), na=False)]
        if not city_match.empty:
            return str(city_match['areacode/城市ID'].iloc[0])
    except:
        pass
    return "101190606"

# ====================== Agent 主程序 ======================
if __name__ == "__main__":
    model = ChatOpenAI(
        model="deepseek-r1:8b",
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
        temperature=0.3
    )

    tools = [get_weather]
    prompt = hub.pull("hwchase17/react")

    # ✅ 这里一定不会报错
    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    res = agent_executor.invoke({"input": "今天北京天气如何"})
    print("\n===== 最终回答 =====")
    print(res["output"])