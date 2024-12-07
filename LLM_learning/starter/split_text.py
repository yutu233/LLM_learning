# 该文件用于分割文本，以便获得token列表进行训练

import re # 正则表达式库

text = "Hello, world, This, is a test."
result = re.split(r'([,.]|\s)', text)
# print(result)

"""
对以上代码的解释:
1.re库: 正则表达式库，用于在字符串中查找和替换文本。
2.re.split(r'([,.]|\s)', text): 正则表达式匹配模式，括号内的字符表示分隔符，这里表示空格。
3.r'([,.]|\s)': 正则表达式匹配模式，括号内的字符表示分隔符，这里表示逗号、句号和空格。
"""

# 上述代码依旧有一些问题，我们没有很好地将空格分离
# 下面的代码将修正这个问题

result = [item.strip() for item in result if item.strip()]
print(result)

"""
对上述代码解释:
1.[item.strip() for item in result if item.strip()]:列表推导式，用于去除列表中每个元素的空格。
    将列表中的每个元素都调用strip()方法，去除空格。
"""

# 继续优化分词, 使其能够处理更多标点符号
text = "Hello, world, Is this -- a test?"
result = re.split(r'([,.?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

"""
对上述代码解释:
1.re.split(r'([,.?_!"()\']|--|\s)', text): 正则表达式匹配模式，括号内的字符表示分隔符，这里表示逗号、句号、问号、下划线、感叹号、双引号、单引号、括号、横杠和空格。
2.result = [item.strip() for item in result if item.strip()]:列表推导式，用于去除列表中每个元素的空格。
3.strip(): 去除字符串两端的空格。
"""