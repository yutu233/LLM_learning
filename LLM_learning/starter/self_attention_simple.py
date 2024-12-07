import torch
from GPT.self_attention import *

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
)

# 计算中间变量ω
# 我们通过计算点积来确定每个输入向量与其他输入向量之间的得分
query = inputs[1]
attention_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x_i, query)

print(attention_scores_2)

"""
参数解释:
inputs: 输入张量，形状为(batch_size, input_dim), batch_size表示批大小，input_dim表示输入维度。
query: 注意力查询向量，形状为(input_dim,)。
attention_scores_2: 注意力得分，形状为(batch_size,)。
torch.empty(inputs.shape[0]): 注意力得分张量，形状为(batch_size,)。
for i, x_i in enumerate(inputs): 遍历输入张量, i表示第i个输入向量的索引，x_i表示第i个输入向量。
enumerate(inputs): 返回一个enumerate对象，包含两个元素，第一个元素是索引，第二个元素是输入张量的第i个元素。
torch.dot(x_i, query): 计算输入向量x_i与查询向量query的点积。
"""
# 点积越高，表示两个向量的对齐程度或相似度越高
# 在自注意力机制中，点积用来衡量序列中个元素之间的关注程度

# 将计算出的分数进行归一化处理, 得到注意力权重α21到α2T
attention_weights_2_temp = attention_scores_2 / attention_scores_2.sum()

print("Attention weights: ", attention_weights_2_temp)
print("Sum: ", attention_weights_2_temp.sum())

"""
参数解释:
atention_weights_2_temp: 归一化后的注意力权重，形状为(batch_size,)
attention_scores_2.sum(): 注意力得分张量的和，表示所有输入向量的注意力得分之和
"""

# 对上述代码的优化: 使用softmax函数进行归一化可以使在处理极端值时表现更佳
def softmax_native(x):
    """
    自定义softmax函数
    """
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attention_weights_2_native = softmax_native(attention_scores_2)

print("Attention weights: ", attention_weights_2_native)
print("Sum: ", attention_weights_2_native.sum())

"""
参数解释:
softmax_native: 自定义softmax函数
attention_weights_2_native: 自定义softmax函数计算的归一化后的注意力权重，形状为(batch_size,)
torch.exp(x): 计算e的x次幂
torch.exp(x).sum(dim=0): 对第0维度求和，表示所有输入向量的注意力得分之和
dim=0: 表示对第0维度求和, 即对batch_size求和
"""

# 实际使用中推荐直接调用PyTorch的softmax函数
attention_weights_2 = torch.softmax(attention_scores_2, dim=0)

print("Attention weights: ", attention_weights_2)
print("Sum: ", attention_weights_2.sum())

# 将嵌入的输入Token x(i)与相应的注意力权重相乘，然后将结果向量求和，计算出上下文向量z(2)
query = inputs[1] # 2nd input tokens is the query
context_vector_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vector_2 += attention_weights_2[i] * x_i

print(context_vector_2)

"""
参数解释:
query: 注意力查询向量，形状为(input_dim,)。
context_vector_2: 上下文向量，形状为(input_dim,)。
torch.zeros(query.shape): 上下文向量张量，形状为(input_dim,)。
"""

# 下面将计算所有上下文向量
attention_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i, j] = torch.dot(x_i, x_j)

print(attention_scores)

"""
参数解释:
attention_scores: 注意力得分矩阵，形状为(batch_size, batch_size)。
torch.empty(6, 6): 注意力得分矩阵，形状为(6, 6)。
atrention_scores[i, j]: 第i行第j列的元素表示第i个输入向量与第j个输入向量之间的注意力得分。
"""

# 在python中, for循环通常较慢
# 因此, 我们可以使用矩阵乘法来计算注意力权重
attention_scores = inputs @ inputs.T

print(attention_scores)

"""
参数解释:
inputs @ inputs.T: 矩阵乘法计算注意力得分矩阵，形状为(batch_size, batch_size)。
@: 矩阵乘法运算符。
"""

# 归一化
attention_weights = torch.softmax(attention_scores, dim=1)

print(attention_weights)

"""
参数解释:
dim=1: 表示对第1维度求softmax, 即对batch_size求softmax。
"""

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])

print("Row 2 sum: ", row_2_sum)
print("All row sums: ", attention_weights.sum(dim=1))

all_context_vectors = attention_weights @ inputs

print(all_context_vectors)

print("Previous 2nd context vector: ", context_vector_2)

x_2 = inputs[1] # A
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

"""
参数解释:
x_2: 输入向量，形状为(input_dim,)。
d_in: 输入维度。
d_out: 输出维度。
W_query: 线性变换矩阵，形状为(input_dim, output_dim)。
W_key: 线性变换矩阵，形状为(input_dim, output_dim)。
W_value: 线性变换矩阵，形状为(input_dim, output_dim)。
torch.manual_seed(123): 设置随机种子。
torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False): 创建参数张量，requires_grad=False表示不需要计算梯度。
torch.rand(d_in, d_out): 随机初始化参数张量。
query_2: 输入向量经过W_query线性变换后的结果，形状为(output_dim,)。
key_2: 输入向量经过W_key线性变换后的结果，形状为(output_dim,)。
value_2: 输入向量经过W_value线性变换后的结果，形状为(output_dim,)。
"""

keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

# 计算注意力得分
keys_2 = keys[1]
attention_score_22 = query_2.dot(keys_2)

print(attention_score_22)

attention_scores_2 = query_2 @ keys.T

print(attention_scores_2)

# 通过除以键的嵌入维度的平方根来缩放注意力得分
d_k = keys.shape[-1]
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)

print(attention_weights_2)

"""
参数解释:
d_k: 键的嵌入维度。
shape[-1]: 最后一维的大小。
dim=-1: 表示对最后一维求softmax。
"""

context_vector_2 = attention_weights_2 @ values

print(context_vector_2)

# an example of the class SelfAttentionV1 in the file "self_attention.py"
torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)

print(sa_v1(inputs))

# an example of the class SelfAttentionV2 in the file "self_attention.py"
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)

print(sa_v2(inputs))

# 将SelfAttentionV2对象的权重矩阵转移到SelfAttentionV1对象

sa_v1 = sa_v2

print(sa_v1(inputs))