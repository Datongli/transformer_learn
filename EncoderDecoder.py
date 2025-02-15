import math
import torch
from Encoder import *
from Decoder import *
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 source_embed: Embedding, target_embed: Embedding, generator: Generator):
        """
        初始化函数
        :param encoder: 编码器类对象
        :param decoder: 解码器类对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 输出部分的类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def encode(self, source: torch.Tensor, source_mask: torch.Tensor):
        """
        编码器逻辑
        :param source: 源数据
        :param source_mask: 源数据掩码张量
        :return: 经过编码器逻辑之后的结果
        """
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory: torch.Tensor, source_mask: torch.Tensor,
               target: torch.Tensor, target_mask: torch.Tensor):
        """
        解码器逻辑
        :param memory: 编码器的输出
        :param source_mask: 源数据掩码张量
        :param target: 目标数据
        :param target_mask: 目前数据掩码张量
        :return: 经过解码器逻辑之后的结果
        """
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

    def forward(self, source: torch.Tensor, target: torch.Tensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        前向传播函数
        :param source: 源数据
        :param target: 目标数据
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return: 前向传播计算的结果
        """
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)


if __name__ == '__main__':
    pass