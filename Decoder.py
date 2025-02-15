import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from Encoder import *


class DecoderLayer(nn.Module):
    """
    解码器层类
    """
    def __init__(self, size: int, self_attn: MultiHeadedAttention, src_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        """
        初始化函数
        :param size: 词嵌入的维度
        :param self_attn: 多头自注意力机制对象
        :param src_attn: 多头常规注意力机制对象
        :param feed_forward: 前馈全连接对象
        :param dropout: 置零比率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 按照解码器层结构图，使用clones函数克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        前向传播函数
        :param x: 上一层输入的张量
        :param memory: 编码器的语义存储张量
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        :return: 输出结果
        """
        # 第一步让x经历第一个子层，多头自注意力机制
        # 采用target_mask，为了将解码时未来的信息进行遮掩（解码第二个字符时，只能看到第一个字符的信息）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 第二步让x经历第二个子层，常规的注意力机制子层，Q!=K=V
        x = self.sublayer[1](x, lambda x: self.src_attn(memory, x, x, source_mask))
        # 第三步让x经历第三个子层，前馈全连接层
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    解码器类
    """
    def __init__(self, layer: DecoderLayer, N: int):
        """

        :param layer: 解码器层的实例
        :param N: 解码器层的数量
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, source_mask: torch.Tensor,
                target_mask: torch.Tensor):
        """
        前向传播函数
        :param x: 输入张量
        :param memory: 编码器层的输出
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据掩码张量
        :return: 输出张量
        """
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    生成器类
    """
    def __init__(self, d_model, vocab_size):
        """
        初始化
        :param d_model: 词嵌入维度
        :param vocab_size: 词表大小
        """
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数
        :param x: 输入张量
        :return: 输出张量
        """
        return F.log_softmax(self.project(x), dim=-1)


if __name__ == "__main__":
    pass