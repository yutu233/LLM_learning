# 将token从Python字符串转为整数表示,生成tokenID
# 这是将tokenID转换成嵌入向量之前的中间步骤

# 创建一个包含所有独特token的列表, 并按字母顺序排序以确定词汇表大小

import re
import requests
from tokenizer import *

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
response = requests.get(url)
raw_text = response.text
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words) # 词汇表大小
print(vocab_size)

"""
函数以及库解释:
1. re.split()：使用正则表达式将字符串分割成子字符串列表。
    split()所需参数: 正则表达式模式, 用于分割字符串的模式, 以及要分割的字符串。
2. list()：将可迭代对象转换为列表。
3.strip(): 用于移除字符串两端的空白字符。
4.sorted(): 用于对可迭代对象进行排序。
"""

# 创建词汇表并打印前五十个单词

vocab = {token:integer for integer, token
        in enumerate(all_words)} # 词汇表字典
for i, item in enumerate(vocab.items()):
    print(item)
    if i == 50:
        break

"""
token:integer: 将词汇表中的每个单词映射到一个唯一的整数。
enumerate()：用于将可迭代对象中的元素及其索引组成一个序列。
"""


# 测试
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," Mrs.Gisburn
said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

# 添加特殊token, 用于处理未知token和分离文本
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

# 检查
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# testV2
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))