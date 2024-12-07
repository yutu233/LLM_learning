# 导入所需的库
import matplotlib.pyplot as plt
from GELU import GELU
import torch
from torch import nn

# 实例化GELU和ReLU激活函数
gelu, relu = GELU(), nn.ReLU()

# 生成输入数据
x = torch.linspace(-3, 3, 100)

# 计算GELU和ReLU的输出
y_gelu, y_relu = gelu(x), relu(x)

# 设置图形的大小
plt.figure(figsize=(8, 3))

# 绘制GELU和ReLU的激活函数图
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()