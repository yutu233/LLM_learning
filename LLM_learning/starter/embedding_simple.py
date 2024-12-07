import torch
from input_target import *

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)
"""
参数解释:
input_ids: 输入的索引列表，每个索引对应一个词语
vocab_size: 词汇表大小，即词汇表中词语的数量
output_dim: 输出维度，即每个词语的向量表示的维度
torch.manual_seed(123): 设置随机种子为123，以便每次运行程序时，得到相同的结果
embedding_layer: 一个Embedding层，用于将输入的索引转换为向量表示
torch.nn.Embedding(vocab_size, output_dim): 创建一个Embedding层，输入维度为vocab_size，输出维度为output_dim
embedding_layer.weight: 该层的权重矩阵，即每个词语的向量表示
"""

