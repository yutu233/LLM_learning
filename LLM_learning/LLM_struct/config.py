# 本文件指定小型GPT2模型的配置

GPT_CONFIG_124M = {
    "vocab_size": 50527, # 词表大小
    "context_length": 1024, # 上下文长度
    "embedding_dim": 768, # 嵌入维度
    "num_heads": 12, # 多头注意力头数
    "num_layers": 12, # 模型层数
    "drop_rate": 0.1, # dropout率
    "qkv_bias": False, # 查询(Query)-键(Key)-值(Value)线性变换是否有偏置
}