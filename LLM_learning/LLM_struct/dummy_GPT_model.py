# 构建占位符GPT主干

import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):

    def __init__(self, config):
        """
        初始化函数

        Args:
            config (dict): 包含模型配置信息的字典，必须包含以下键值对：
                - "vocab.size" (int): 词汇表大小
                - "embedding_dim" (int): 词嵌入维度
                - "context_length" (int): 上下文长度
                - "drop_rate" (float): Dropout层丢弃率
                - "num_layers" (int): Transformer层数

        Returns:
            None
        """
        super().__init__()

        # 创建词嵌入层
        self.token_emb = nn.Embedding(config["vocab_size"], config["embedding_dim"])

        # 创建位置嵌入层
        self.pos_emb = nn.Embedding(config["context_length"], config["embedding_dim"])

        # 创建Dropout层
        self.drop_emb = nn.Dropout(config["drop_rate"])

        # 创建多个DummyTransformerBlock层组成的序列
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(config) for _ in range(config["num_layers"])])

        # 创建层归一化层
        self.final_norm = DummyLayerNorm(config["embedding_dim"])

        # 创建输出层
        self.out_head = nn.Linear(
            config["embedding_dim"], config["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        """
        前向传播函数。

        Args:
            in_idx (torch.Tensor): 输入的索引张量，shape为(batch_size, seq_len)。

        Returns:
            torch.Tensor: 经过模型计算后的logits张量，shape为(batch_size, seq_len, vocab_size)。

        """
        # 获取输入张量的形状，分别为batch_size和seq_len
        batch_size, seq_len = in_idx.shape
        # 将输入索引张量通过token_emb层得到token嵌入张量
        token_embeds = self.token_emb(in_idx)
        # 根据序列长度生成位置索引，并通过pos_emb层得到位置嵌入张量
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 将token嵌入张量和位置嵌入张量相加得到初始的x张量
        x = token_embeds + pos_embeds
        # 对x张量进行dropout操作
        x = self.drop_emb(x)
        # 将x张量通过trf_blocks层进行模型计算
        x = self.trf_blocks(x)
        # 对x张量进行归一化操作
        x = self.final_norm(x)
        # 将x张量通过out_head层得到logits张量
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    
    def __init__(self, config):
        """
        初始化函数，用于创建类的实例对象。
        
        Args:
            config (dict): 包含配置信息的字典，用于初始化类的实例对象。
        
        Returns:
            None
        
        """
        super().__init__()
    
    def forward(self, x):
        """
        对输入张量x进行前向传播，直接返回输入张量x。
        
        Args:
            x (torch.Tensor): 输入张量，可以是任意形状和数据类型的张量。
        
        Returns:
            torch.Tensor: 返回与输入张量x形状和数据类型相同的张量。
        
        """
        return x
    
class DummyLayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x