import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from pyexpat import features
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


def clones(module: any, N: int):
    """
    克隆函数，因为在多头注意力机制的实现中，用到多个结构相同的模型层
    静态函数
    :param module: 模型层
    :param N: 要克隆的数量
    :return: nn.ModuleList类型的模型层列表
    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


class MultiHeadedAttention(nn.Module):
    def __init__(self, head: int, embedding_dim: int, dropout: float=0.1):
        """
        类初始化函数
        :param head: 多头注意力机制，注意力头的数量
        :param embedding_dim: 词填充维度
        :param dropout: 抛弃层参数
        """
        super(MultiHeadedAttention, self).__init__()
        # 使用测试中常用的assert语句，判断词填充维度是否可以被头数整除
        assert embedding_dim % head == 0
        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 传入头数
        self.head = head
        # 获得线性层对象，内部是4个embedding_dim x embedding_dim矩阵
        # 4个的原因是，多头注意力机制中，Q,K,V各需要一个，在最后拼接的时候还需要一个
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.attn代表最后得到的注意力张量，初始化为None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None, dropout: nn.Dropout = None):
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
        # 多头处理环节
        # 利用zip将self.linears和(query, key, value)组合在一起，循环按照小的次数来（3次）
        # 每一次取出的事self.linears中的一个线性层和(query, key, value)中的一个
        # 在完成线性变换后，开始为每个头分割输入，利用view方法对线性变换的结果进行维度重塑
        # 对第二维和第三维进行转置，每个头可以获得一部分词特征组成的句子
        # 这样代表句子长度的维度（-1）和代表词向量的维度（self.d_k）可以相邻，注意力机制更加方便找到词义和句子位置的关系
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]
        # 得到每个头的输入后，传入attention中
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将计算得到的张量转换为和输入同样维度的张量
        # 由于先对x进行了transpose操作，所以必须紧跟着进行contiguous操作，才可以进行view操作
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)
        # 调用最后一个线性层作为多头注意力的输出
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    前馈全连接层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化函数
        :param d_model: 词嵌入维度
        :param d_ff: 中间节点数
        :param dropout: 置零比例
        """
        super(PositionwiseFeedForward, self).__init__()
        # 实例化两个线性层对象
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)  # 非线性激活
        x = self.dropout(x)
        x = self.w2(x)
        return x


class LayerNorm(nn.Module):
    """通过LayerNorm实现规范化层的类"""
    def __init__(self, features, eps=1e-6):
        """
        初始化函数
        :param features: 词嵌入维度
        :param eps: 一个比较小的数，防止0在分母上出现
        """
        super(LayerNorm, self).__init__()
        # 初始化两个参数
        # 使用nn.Parameter封装，表示需要跟着模型进行训练
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        计算主函数
        :param x: 输入的张量
        :return: 规范化之后的张量
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    """
    子层连接结构的类
    """
    def __init__(self, size, dropout=0.1):
        """
        初始化函数
        :param size: 词嵌入维度的大小
        :param dropout: 置零比例
        """
        super(SublayerConnection, self).__init__()
        # 实例化层归一化（注意和批量归一化BatchNorm的区别）
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer):
        """
        前向传播函数
        :param x: 输入
        :param sublayer: 连接中的子层函数
        :return: 前向传播的计算结果
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    使用EncoderLayer类实现编码器层
    """
    def __init__(self, size: int, self_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        """
        初始化函数
        :param size: 词嵌入维度的大小
        :param self_attn: 多头注意力子层的实例化对象
        :param feed_forward: 前馈全连接层的实例化对象
        :param dropout: 置零比例
        """
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 由于编码器层中有两个子层连接结构，所以用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        前向传播函数
        :param x: 输入张量
        :param mask: 掩码张量
        :return: 输出张量
        """
        # 第一个子层，多头注意力机制层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层，前馈全连接层
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    编码器类
    """
    def __init__(self, layer: EncoderLayer, N):
        """
        初始化函数
        :param layer: 子层的类
        :param N: N个子层
        """
        super(Encoder, self).__init__()
        # 克隆N个编码器层
        self.layers = clones(layer, N)
        # 初始化一个规范化层，用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        前向传播函数
        :param x: 输入张量
        :param mask: 掩码张量
        :return: 输出张量
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



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
    attn, p_attn = MultiHeadedAttention.attention(query, key, value)  # 使用类的静态方法
    print("attn:", attn)
    print("p_attn:", p_attn)
    head, embedding_dim, dropout = 8, 512, 0.2
    mask = Variable(torch.zeros(1, 8, 4, 4))
    mha = MultiHeadedAttention(head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)
    print(mha_result)
    print(mha_result.shape)
    d_model, d_ff, dropout = 512, 64, 0.2
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_result = ff(mha_result)
    print("ff_result:", ff_result)
    print("shape of ff_result:", ff_result.shape)
    features = d_model = 512
    eps = 1e-6
    ln = LayerNorm(features, eps)
    ln_result = ln(ff_result)
    print(ln_result)
    print(ln_result.shape)
    size = d_model = 512
    head = 8
    dropout = 0.2
    mask = Variable(torch.zeros(1, 8, 4, 4))
    self_attn = MultiHeadedAttention(head, d_model)
    sublayer = lambda x: self_attn(x, x, x, mask)
    sc = SublayerConnection(size, dropout)
    sc_result = sc(pe_result, sublayer)
    print("sc_result:", sc_result)
    print("shape of sc_result:", sc_result.shape)
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    x = pe_result
    dropout = 0.2
    self_attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(1, 8, 4, 4))
    el = EncoderLayer(size, self_attn, ff, dropout)
    el_result = el(x, mask)
    print(el_result)
    print(el_result.shape)
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    dropout = 0.2
    layer = EncoderLayer(size, c(attn), c(ff), dropout)
    N = 8
    mask = Variable(torch.zeros(1, 8, 4, 4))
    en = Encoder(layer, N)
    en_result = en(x, mask)
    print(en_result)
    print(en_result.shape)