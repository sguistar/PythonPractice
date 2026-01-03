import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# K(=Q)和V的维度
d_k = 64
d_v = 64


# 定义缩放点积注意力
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2) / np.sqrt(d_k))
        scores.masked_fill_(attn_mask, -1e9)
        weights = F.softmax(scores, dim=-1)  # 或 nn.Softmax(dim=-1)(scores)
        output_context = torch.matmul(weights, v)
        return output_context, weights


d_embedding = 512  # Embedding 的维度
n_heads = 8  # Multi-Head Attention 中头的个数
batch_size = 3  # 每一批的数据大小


# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)
        self.linear = nn.Linear(d_v * n_heads, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, q, k, v, attn_mask=None):
        residual, batch_size = q, q.size(0)

        q_s = self.W_Q(q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(k).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(v).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) if attn_mask is not None else None
        context, weights = ScaleDotProductAttention()(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        output = self.layer_norm(context + residual)
        output = self.linear(output)
        return output, weights


# 定义逐位置前馈网络
class PositionFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PositionFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        residual = inputs  # 保留残差块
        output = F.relu(self.conv1(inputs.transpose(1, 2)))
        output = F.relu(self.conv2(output)).transpose(1, 2)
        output = self.layer_norm(output + residual)
        return output


# 生成正弦位置编码表的函数，用于在 Transformer 中引入位置信息
def get_sin_enc_table(n_position, embedding_dim):
    # 根据位置和维度信息，初始化正弦位置编码表
    sin_enc_table = torch.zeros((n_position, embedding_dim))
    # 遍历所有位置和维度，计算角度值
    for i in range(n_position):
        for j in range(embedding_dim):
            angle = i / np.power(10000, 2 * (j // 2) / embedding_dim)
            sin_enc_table[i, j] = angle

    # 计算正弦和余弦值
    sin_enc_table[:, 0::2] = torch.sin(sin_enc_table[:, 0::2])  # dim 2i 偶数维
    sin_enc_table[:, 1::2] = torch.cos(sin_enc_table[:, 1::2])  # dim 2i+1 奇数维

    return torch.FloatTensor(sin_enc_table)


# 定义填充注意力掩码函数
def get_attn_pad_mask(seq_q, seq_k):
    mask_q = seq_q.data.ne(0).unsqueeze(2)
    mask_k = seq_k.data.ne(0).unsqueeze(2)
    # 构建掩码矩阵
    valid_encoder_pos_matrix = torch.bmm(mask_q.float(), mask_k.transpose(2, 1).float())
    invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
    pad_attn_mask = invalid_encoder_pos_matrix.to(torch.bool)

    return pad_attn_mask


# 定义编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 多头注意力层
        self.pos_ffn = PositionFeedForwardNet()  # 位置前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn_weights = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, attn_mask=enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn_weights


# 定义编码组件
n_layers = 6  # 设置 Encoder 的层数


class EncoderComponent(nn.Module):
    def __init__(self, corpus):
        super(EncoderComponent, self).__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding)
        self.pos_emb = nn.Embedding.from_pretrained(embeddings=get_sin_enc_table(corpus.src_len + 1, d_embedding),
                                                    freeze=True)
        self.layers = nn.ModuleList([Encoder() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1, device=enc_inputs.device)
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        enc_pad_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_weights = []
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(
                enc_outputs, enc_pad_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)

        return enc_outputs, enc_self_attn_weights


# 生成后续注意力掩码的函数，用于在多头自注意力计算中忽略未来信息
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  # 获取输入序列的形状
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)  # 使用 numpy 创建一个上三角矩阵
    # 将 numpy 数组转换为 PyTorch 张量，并将数据类型设置为 byte（布尔值）
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    # 确保张量在与输入相同的设备上
    subsequent_mask = subsequent_mask.to(seq.device)
    return subsequent_mask


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_pad_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        attn_mask=dec_self_attn_mask)
        dec_outputs, dec_enc_attn_weight = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                             attn_mask=dec_enc_pad_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn_weight


n_layer = 6


class DecoderComponent(nn.Module):
    def __init__(self, corpus):
        super(DecoderComponent, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding)
        self.pos_emb = nn.Embedding.from_pretrained(embeddings=get_sin_enc_table(corpus.tgt_len + 1, d_embedding),
                                                    freeze=True)
        self.layers = nn.ModuleList([Decoder() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        pos_indices = torch.arange(1, dec_inputs.size(
            1) + 1, device=dec_inputs.device).unsqueeze(0)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)

        # 生成解码器自注意力掩码和解码器 - 编码器注意力掩码
        dec_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # 填充位掩码
        dec_subsequent_mask = get_attn_subsequent_mask(dec_inputs)  # 后续位掩码
        dec_self_attn_mask = torch.gt(
            (dec_pad_mask + dec_subsequent_mask), 0)  # 解码器自注意力掩码
        dec_enc_pad_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs)  # 解码器-编码器填充掩码

        # 生成解码器自注意力和解码器-编码器注意力权重
        dec_self_attns, dec_enc_attn_weights = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn_weight = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                                    dec_enc_pad_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attn_weights.append(dec_enc_attn_weight)

        return dec_outputs, dec_self_attns, dec_enc_attn_weights


class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = EncoderComponent(corpus)
        self.decoder = DecoderComponent(corpus)
        self.projection = nn.Linear(
            d_embedding, len(corpus.tgt_vocab), bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attn_weights = self.encoder(enc_inputs)
        dec_outputs, dec_self_attn_weights, dec_enc_attn_weights = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)

        return dec_logits, enc_self_attn_weights, dec_self_attn_weights, dec_enc_attn_weights

# 使用 NMTCorpus 类替代旧的 TranslationCorpus 类
# NMTCorpus 类在 train.py 文件中定义
