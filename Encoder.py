import torch
import torch.nn as nn
import math
from torch.autograd import Variable  # torch中变量封装函数Variable


class Embedding(nn.Module):
    """
    文本嵌入层，将文本的数字表示转化为向量表示
    继承自nn.Module，这样就有标准层的一些功能
    """
    def __init__(self, d_model, vocab):
        """
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embedding, self).__init__()
        # 调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
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
    def __init__(self, d_model, dropout, max_len=5000):
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

    def forward(self, x):
        """forward函数的参数是x，表示文本序列的词嵌入表示"""
        # 在相加之前需要对pe做一些适配工作，将这个三维张量的第二维也就是句子最大长度那一维将切片到与输入的x的第二维相同
        # 因为我们默认max_len为5000一般来讲实在是太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配
        # 然后使用Variable封装，将requires_grad设置为False，因为不需要对pe进行梯度下降
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


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