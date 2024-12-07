# 导入所需的库
import tiktoken
import torch
from dummy_GPT_model import *
from config import GPT_CONFIG_124M
from layer_norm import *
from feed_forward import FeedForward
from shortcut_connection import ExampleDeepNeuralNetwork

# 获取GPT-2模型的编码器
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# 将文本编码为张量并添加到批处理中
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
# 将批处理中的张量堆叠在一起
batch = torch.stack(batch, dim=0)

# print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)

# print("Output shape:", logits.shape)
# print(logits)

# 设置随机种子以确保结果可重复
torch.manual_seed(123)

# 创建一个形状为 (2, 5) 的随机张量，代表一批输入样本
batch_example = torch.randn(2, 5)

# 创建一个包含线性层和ReLU激活函数的顺序神经网络
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

# 通过神经网络层传递输入样本，获得输出结果
out = layer(batch_example)

# 打印输出结果
# print(out)

# 计算输出张量的均值和方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

# 打印均值和方差
# print("Mean:\n", mean)
# print("Variance:\n", var)

# 对层输出进行标准化，计算均值和方差
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)

# 打印标准化后的层输出、均值和方差
# print("Normalized layer outputs:\n", out_norm)
# print("Mean:\n", mean)
# print("Variance:\n", var)

torch.set_printoptions(sci_mode=False)
# print("Mean:\n", mean)
# print("Variance:\n", var)

# 实例化一个LayerNorm对象
ln = LayerNorm(emb_dim=5)

# 对batch_example进行层归一化处理
out_ln = ln(batch_example)

# 计算处理后输出的均值
mean = out_ln.mean(dim=-1, keepdim=True)

# 计算处理后输出的方差
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)

# print("Mean:\n", mean)
# print("Variance:\n", var)

# FeedForward类的实例化，用于创建前馈神经网络
ffn = FeedForward(GPT_CONFIG_124M)

# 生成一个形状为(2, 3, 768)的随机张量，作为输入
x = torch.rand(2, 3, 768)

# 通过前馈神经网络处理输入张量
out = ffn(x)

# 输出结果的形状
# print(out.shape)

# 定义网络层的尺寸
layer_sizes = [3, 3, 3, 3, 3, 1]
# 创建一个示例输入
sample_input = torch.tensor([[1., 0., -1.]])
# 设置随机种子以确保可复现性
torch.manual_seed(123)

# 实例化一个不使用快捷连接的深度神经网络
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

def print_grandients(model, x):
    """
    计算模型梯度并打印权重参数的梯度均值。
    
    Args:
        model (torch.nn.Module): 待计算梯度的模型。
        x (torch.Tensor): 模型的输入数据。
    
    Returns:
        None
    
    """
    # 将输入数据传入模型，得到输出
    output = model(x)
    # 定义目标值为一个包含单个元素的张量
    target = torch.tensor([[0.]])
    # 定义均方误差损失函数
    loss = nn.MSELoss()
    # 计算损失值
    loss = loss(output, target)

    # 反向传播，计算梯度
    loss.backward()

    # 遍历模型的命名参数
    for name, param in model.named_parameters():
        # 如果参数名中包含'weight'
        if 'weight' in name:
            # 打印参数名及其梯度绝对值的均值
            print(f"{name} has gradient mean of {param.grad.abs().mean()}")

# print_grandients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)

print_grandients(model_with_shortcut, sample_input)
