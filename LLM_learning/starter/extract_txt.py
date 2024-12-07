# 本文件用于提取为大语言模型训练的文本数据
# 以伊迪斯·沃顿的短篇小说《判决》为例

import requests # 导入requests模块用于下载网页
import re

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
response = requests.get(url) # 下载网页内容
raw_text = response.text # 获取网页内容
print("Total number of characters: ", len(raw_text)) # 打印文本长度
print(raw_text[:99])# 打印前100个字符

"""
对以上代码的解释:
1.requests模块: 用于发送HTTP请求，获取网页内容
2. get函数: 用于发送GET请求，获取网页内容
3. response.text: 获取网页内容的字符串形式
4. len函数: 计算字符串长度
"""

# 将分词器部署
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text) # 使用正则表达式对文本进行分词
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) # 打印分词后的文本长度
# 这是该文本中的token数量(不包括空格)

# 打印前30个以进行快速目测
print(preprocessed[:30])