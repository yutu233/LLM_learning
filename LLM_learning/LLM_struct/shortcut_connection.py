import torch
import torch.nn as nn
from torch.nn import GELU

class ExampleDeepNeuralNetwork(nn.Module):

    def __init__(self, layer_sizes, use_shortcut: bool):
        """
        初始化一个五层神经网络。

        Args:
            layer_sizes (list[int]): 神经网络各层的神经元个数，列表长度为6，分别表示输入层、中间三层、输出层的神经元个数。
            use_shortcut (bool): 是否使用残差连接。

        Returns:
            None
        """
        super().__init__()
        self. use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            # 实现五层神经网络
            # 初始化第一层神经网络，输入层到第一层隐藏层
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            # 初始化第二层神经网络，第一层隐藏层到第二层隐藏层
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            # 初始化第三层神经网络，第二层隐藏层到第三层隐藏层
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            # 初始化第四层神经网络，第三层隐藏层到第四层隐藏层
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            # 初始化第五层神经网络，第四层隐藏层到输出层
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            # 遍历每一层
            # 计算当前层输出
            layer_output = layer(x)
            # 获取当前层的输出
            # 检查是否可以使用快捷连接
            if self.use_shortcut and x.shape == layer_output.shape:
                # 如果启用了快捷连接，并且当前输入和当前层输出形状相同
                x = x + layer_output
            # 则将当前输入和当前层输出相加
            else:
                x = layer_output
            # 否则直接将当前层输出作为下一层的输入
        return x