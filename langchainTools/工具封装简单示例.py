from langchain.tools import tool

# 封装：只需要加 @tool 装饰器
@tool
def add(a: int, b: int) -> int:
    """
    两个数字相加
    必传参数：a (数字), b (数字)
    """
    return a + b

# 使用：直接像函数一样调用
print(add.invoke({"a": 10, "b": 20}))  # 输出 30