import requests
import pandas as pd
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """调用实时天气API返回温度及天气状况，参数city为中国城市名称，如：北京、上海、广州"""
    city_id = get_city_id(city)
    url = "https://eolink.o.apispace.com/456456/weather/v001/now"
    payload = {"areacode": city_id}
    headers = {"X-APISpace-Token": "xnkahfw6hr30g011r2tcqf6w9q9p9yx8"}
    response = requests.get(url, params=payload, headers=headers)
    res = response.json()
    try:
        realtime = res["result"]["realtime"]
        city_name = res["result"]["location"]["name"]
        weather = realtime["text"]
        temp = realtime["temp"]
        return f"{city_name}：{weather}，气温 {temp} ℃"
    except Exception as e:
        return f"天气数据解析失败：{str(e)}"


def get_city_id(city: str) -> str:
    """根据城市名称模糊查询城市id"""
    city_df = pd.read_csv("Sheet1.csv")
    # 优先匹配区县
    district_match = city_df[city_df["district"].str.contains(city.strip())]
    if not district_match.empty:
        return district_match["areacode/城市ID"].iloc[0]
    # 匹配城市
    city_match = city_df[city_df["city"].str.contains(city.strip())]
    if not city_match.empty:
        return city_match["areacode/城市ID"].iloc[0]
    return "101190606"  # 默认


if __name__ == "__main__":
    print(get_weather.invoke("北京"))