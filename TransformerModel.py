from EncoderDecoder import *


def make_model(source_vocab: int, target_vocab: int, N: int=6, d_model: int=512,
               d_ff: int=2048, head: int=8, dropout: float=0.1):
    """
    构建transformer模型的函数（示例是用于翻译任务，例如英译汉）
    :param source_vocab: 源数据词汇（特征）总数
    :param target_vocab: 目标数据词汇（特征）总数
    :param N: 编码器和解码器堆叠数
    :param d_model: 词嵌入维度
    :param d_ff: 前馈全连接网络中中间层的维度
    :param head: 多头注意力机制中头的数目
    :param dropout: 置零比率
    :return:
    """
    # 得到一个深度拷贝命令，因为接下来许多结构都需要进行深度拷贝
    c = copy.deepcopy
    # 实例化多头注意力类
    attn = MultiHeadedAttention(head, d_model)
    # 实例化前馈全连接类
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码器
    position = PositionalEncoding(d_model, dropout)
    # 构建模型
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embedding(d_model, source_vocab), c(position)),
                           nn.Sequential(Embedding(d_model, target_vocab), c(position)),
                           Generator(d_model, target_vocab)
                           )
    # 如果参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == '__main__':
    source_vocab = 11
    target_vocab = 11
    res = make_model(source_vocab, target_vocab)
    print(res)