import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable  # torch中变量封装函数Variable
import copy


class Embedding(nn.Module):
    """
    文本嵌入层，将文本的数字表示转化为向量表示
    继承自nn.Module，这样就有标准层的一些功能
    """
    def __init__(self, d_model: int, vocab: int):
        """
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embedding, self).__init__()
        # 调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        """
        :param x: 输入给模型的文本通过词汇映射后的张量
        """
        # 前向传播逻辑
        # 将x传递给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)  # 有一个缩放的作用


class PositionalEncoding(nn.Module):
    """
    位置编码器
    将词汇位置不同产生的信息加入到词嵌入张量中
    """
    def __init__(self, d_model: int, dropout: float, max_len: int|float=5000):
        """
        :param d_model: 特征维度
        :param dropout: 置0比例
        :param max_len: 每个句子最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 实例化nn中预定义的Dropout层，并将dropout传入其中
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，max_len行，d_model列
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，词汇的绝对位置用其索引表示
        # 用arrange方法获得一个连续自然数向量，再使用unsqueeze方法扩展向量维度
        # 最终形成一个max_len x 1 的向量矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中
        # 可以将max_len x 1的绝对位置矩阵扩展为max_len x d_model的矩阵
        # 希望将自然数绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快收敛
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 填充位置编码矩阵的偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 填充位置编码矩阵的奇数列
        # 扩展一个维度
        pe = pe.unsqueeze(0)  # 扩展维度，形成1 x max_len x d_model的张量
        # 将pe位置编码矩阵注册成模型的buffer
        # buffer：认为对模型效果有帮助，但是不属于模型的参数或者超参数，不需要进行优化训练
        # 注册之后可以在模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """forward函数的参数是x，表示文本序列的词嵌入表示"""
        # 在相加之前需要对pe做一些适配工作，将这个三维张量的第二维也就是句子最大长度那一维将切片到与输入的x的第二维相同
        # 因为我们默认max_len为5000一般来讲实在是太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配
        # 然后使用Variable封装，将requires_grad设置为False，因为不需要对pe进行梯度下降
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size: int):
    """
    生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它的最后两维形成一个方阵
    :param size: 矩阵的大小
    :return: 掩码张量
    """
    # 定义掩码张量的形状
    attn_shape = (1, size, size)
    # 形成上三角矩阵，同时将其中数据类型转化为无符号整型节约空间
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 进行行列翻转操作形成下三角阵（1为看到，0为看不到），同时转化为张量
    return torch.from_numpy(1 - subsequent_mask)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None, dropout: nn.Dropout=None):
    """
    注意力机制的实现
    :param query:查询
    :param key:键
    :param value:值
    :param mask:掩码张量
    :param dropout:抛弃层的失活率
    :return:注意力表示和注意力张量
    """
    # query的最后一维度的大小，一般情况下等同于词嵌入维度
    d_k = query.size(-1)
    # 按照注意力公式计算注意力张量
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 注意力得分张量
    # 判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法，将掩码张量和scores张量的每一位置比较
        # 如果掩码张量在相应位置上是0，则用-1e9这个很小的值来替代
        scores = scores.masked_fill(mask == 0, -1e9)
    # 在scores的最后一个维度上进行softmax操作
    p_attn = F.softmax(scores, dim=-1)
    # 判断是否使用dropout
    if dropout is not None:
        dropout(p_attn)
    # 最后完成p_attn和value张量的乘法，并返回query的注意力表示
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, head: int, embedding_dim: int, dropout: float=0.1):
        """
        类初始化函数
        :param head: 多头注意力机制，注意力头的数量
        :param embedding_dim: 词填充维度
        :param dropout: 抛弃层参数
        """
        super(MultiHeadedAttention).__init__()
        # 使用测试中常用的assert语句，判断词填充维度是否可以被头数整除
        assert embedding_dim % head == 0
        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 传入头数
        self.head = head
        # 获得线性层对象，内部是4个embedding_dim x embedding_dim矩阵
        # 4个的原因是，多头注意力机制中，Q,K,V各需要一个，在最后拼接的时候还需要一个
        self.linears = self.clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.attn代表最后得到的注意力张量，初始化为None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    @staticmethod
    def clones(module: any, N: int):
        """
        克隆函数，因为在多头注意力机制的实现中，用到多个结构相同的模型层
        静态函数
        :param module: 模型层
        :param N: 要克隆的数量
        :return: nn.ModuleList类型的模型层列表
        """
        return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None):
        """
        前向传播函数
        :param query: query
        :param key: key
        :param value: value
        :param mask: 掩码张量
        :return:
        """
        # 如果掩码张量非空
        if mask is not None:
            # 扩展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)
        # 获得一个训练批次中的样本数量
        batch_size = query.size(0)



if __name__ == '__main__':
    # 词嵌入的维度是512维
    d_model = 512
    # 词表的大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量，形状为2x4
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    # 调用
    emb = Embedding(d_model, vocab)
    embr = emb(x)
    print("embr: ", embr)
    dropout = 0.1
    max_len = 60
    x = embr
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)
    print("pe_result: ", pe_result)
    print("pe_result.shape: ", pe_result.shape)
    size = 5
    mask = subsequent_mask(size)
    print("mask: ", mask)
    query = key = value = pe_result
    attn, p_attn = attention(query, key, value)
    print("attn:", attn)
    print("p_attn:", p_attn)
