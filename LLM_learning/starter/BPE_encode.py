
from importlib.metadata import version # 检查版本
import tiktoken
import re
"""
print("tiktoken version: ", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2") # 初始化BPE分词器
text = "Hello, do you like tea? <|endoftext|> In the sunlit terrace of some"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) # 编码文本, 添加分词符

print(integers)

strings = tokenizer.decode(integers) # 解码文本

print(strings)
"""

tokenizer = tiktoken.get_encoding("gpt2")

with open(r"C:\VSCODE\LLM_learning\resources\the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
encode_text = tokenizer.encode(raw_text)
print(len(encode_text))

encode_sample = encode_text[50:] # 展示50个token之后的内容
context_size= 4 # A
x = encode_sample[:context_size]
y = encode_sample[1:context_size  + 1]

print(f"x: {x}")
print(f"y:  {y}")

for i in range(1, context_size + 1):
    context = encode_sample[:i]
    desired = encode_sample[i]
    print(context, "--->", desired)

for i in range(1, context_size + 1):
    context = encode_sample[:i]
    desired = encode_sample[i]
    print(tokenizer.decode(context), "-->", tokenizer.decode([desired]))
    