import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import Counter

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

        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, n_heads, 1, 1) if attn_mask is not None else None
        context, weights = ScaleDotProductAttention()(
            q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, n_heads * d_v)

        output = self.layer_norm(context + residual)
        output = self.linear(output)
        return output, weights


# 定义逐位置前馈网络
class PositionFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PositionFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
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
    mask_q = seq_q.data.ne(0).to(torch.int32).unsqueeze(2)
    mask_k = seq_k.data.ne(0).to(torch.int32).unsqueeze(2)
    # 构建掩码矩阵
    valid_encoder_pos_matrix = torch.bmm(mask_q, mask_k.transpose(2, 1))
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
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1)
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
            1) + 1).unsqueeze(0).to(dec_inputs)
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


text = [['小冰 喜欢 音乐', 'XiaoBing likes music'], ['我 爱 学习 人工智能', 'I love studying AI'], ['深度学习 改变 世界', 'DL changed the world'], ['自然语言处理 很 强大', 'NLP is powerful'], ['神经网络 非常 复杂', 'Neural-networks are complex'], ['我 喜欢 编程', 'I like coding'], ['他 在 学习 Python', 'He is learning Python'], ['我们 要 吃 午饭', 'We are going to eat lunch'], ['今天 天气 很 好', 'The weather is nice today'], ['她 爱 喝 咖啡', 'She loves drinking coffee'], ['猫 很 可爱', 'Cats are cute'], ['狗 非常 忠诚', 'Dogs are very loyal'], ['我要 去 旅行', 'I want to travel'], ['暑假 想 学吉他', 'I want to learn guitar this summer'], ['数据 科学 很 有趣', 'Data science is interesting'], ['机器 学习 需要 数据', 'Machine learning needs data'], ['模型 训练 花 时间', 'Training models takes time'], ['算法 需要 优化', 'Algorithms need optimization'], ['他 在 写 论文', 'He is writing a paper'], ['研究 团队 在 合作', 'The research team is collaborating'], ['开会 时间 已 确定', 'Meeting time is fixed'], ['请 提交 报告', 'Please submit the report'], ['我 在 看 书', 'I am reading a book'], ['她 在 听 音乐', 'She is listening to music'], ['孩子 在 玩 球', 'The child is playing ball'], ['今晚 有 篮球 比赛', 'There is a basketball game tonight'], ['我 想 学 画画', 'I want to learn painting'], ['数学 很 有挑战性', 'Math is challenging'], ['线性 代数 很 重要', 'Linear algebra is important'], ['统计 学 帮助 决策', 'Statistics help decision making'], ['概率 论 很 有意思', 'Probability theory is interesting'], ['他 是 一个 好 老师', 'He is a good teacher'], ['请 帮我 调试 代码', 'Please help me debug the code'], ['这个 函数 有 错误', 'This function has a bug'], ['编译 通过 了', 'Compilation succeeded'], ['运行 很 顺利', 'The run was smooth'], ['内存 使用 很 高', 'Memory usage is high'], ['磁盘 空间 不足', 'Disk space is low'], ['网络 连接 丢失', 'Network connection lost'], ['保存 文件 成功', 'File saved successfully'], ['打开 浏览 器', 'Open the browser'], ['提交 pull 请求', 'Submit a pull request'], ['合并 分支 完成', 'Branch merge completed'], ['我 在 学 Git', 'I am learning Git'], ['他 喜欢 下 棋', 'He likes playing chess'], ['周末 去 爬山', 'Go hiking on the weekend'], ['公园 里 有 花', 'There are flowers in the park'], ['城市 很 热闹', 'The city is lively'], ['乡村 很 安静', 'The countryside is quiet'], ['我 想 吃 披萨', 'I want to eat pizza'], ['她 做 了 蛋糕', 'She made a cake'], ['早餐 喝 牛奶', 'Drink milk for breakfast'],
        ['请 多 练习 发音', 'Please practice pronunciation more'], ['学 语言 需 持续', 'Learning a language requires consistency'], ['他 在 学日语', 'He is learning Japanese'], ['我们 在 讨论 项目', 'We are discussing the project'], ['代码 评审 开始 了', 'Code review has started'], ['测试 用例 已 编写', 'Test cases have been written'], ['模型 精度 提高', 'Model accuracy improved'], ['超参数 需要 调整', 'Hyperparameters need tuning'], ['训练 数据 不平衡', 'Training data is imbalanced'], ['请 做 数据 增强', 'Please do data augmentation'], ['保存 模型 权重', 'Save model weights'], ['加载 预训练 模型', 'Load pretrained model'], ['GPU 加速 很 重要', 'GPU acceleration is important'], ['云 平台 更 方便', 'Cloud platforms are more convenient'], ['他 在 调试 神经网', 'He is debugging a neural net'], ['文本 分类 很 有用', 'Text classification is useful'], ['情感 分析 越来越 流行', 'Sentiment analysis is getting popular'], ['推荐 系统 协助 选择', 'Recommendation systems assist choices'], ['搜索 引擎 提高 体验', 'Search engines improve experience'], ['图片 识别 很 实用', 'Image recognition is practical'], ['目标 检测 在 进步', 'Object detection is progressing'], ['语音 识别 在 发展', 'Speech recognition is developing'], ['他 喜欢 看 科幻', 'He likes sci-fi'], ['电影 晚上 一起 看', 'Watch a movie together tonight'], ['她 学会 了 烘焙', 'She learned baking'], ['孩子 爱 画画', 'The kid loves drawing'], ['花园 有 一些 果树', 'The garden has some fruit trees'], ['早晨 跑步 很 舒服', 'Morning runs feel good'], ['晚上 喝 茶 放松', 'Drinking tea at night relaxes'], ['他 在 练 翻译', 'He is practicing translation'], ['我 想 要 一台 笔记本', 'I want a laptop'], ['这 是 一个 好 点子', 'This is a good idea'], ['请 发送 邮件', 'Please send the email'], ['我 需要 更多 样本', 'I need more samples'], ['实验 结果 有 希望', 'Experimental results look promising'], ['调参 花了 很久', 'Tuning took a long time'], ['学习 新 知识 开心', 'Learning new knowledge is fun'], ['团队 合作 很 关键', 'Teamwork is crucial'], ['项目 进展 顺利', 'Project progress is smooth'], ['服务器 重启 完成', 'Server restart completed'], ['数据库 备份 了', 'Database backed up'], ['他 修复 了 缺陷', 'He fixed the bug'], ['功能 已 发布', 'Feature has been released'], ['客户端 更新 了', 'Client updated'], ['界面 更 美观', 'The interface looks nicer'], ['性能 得到 提升', 'Performance improved'], ['代码 文档 完整', 'Code documentation is complete'], ['开源 社区 很 活跃', 'The open-source community is active']]


# 定义 TranslationCorpus 类
class TranslationCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        # 计算源语言和目标语言的最大句子长度，并分别加 1 和 2 以容纳填充符和特殊符号
        self.src_len = max(len(sentence[0].split())
                           for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split())
                           for sentence in sentences) + 2
        # 创建源语言和目标语言的词汇表
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        # 创建索引到单词的映射
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}
    # 定义创建词汇表的函数

    def create_vocabularies(self):
        # 统计源语言和目标语言的单词频率
        src_counter = Counter(
            word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(
            word for sentence in self.sentences for word in sentence[1].split())
        # 创建源语言和目标语言的词汇表，并为每个单词分配一个唯一的索引
        src_vocab = {'<pad>': 0, **
                     {word: i+1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2,
                     **{word: i+3 for i, word in enumerate(tgt_counter)}}
        return src_vocab, tgt_vocab
    # 定义创建批次数据的函数

    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch, target_batch = [], [], []
        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        for index in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[index]
            # 将源语言和目标语言的句子转换为索引序列
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word]
                                                   for word in tgt_sentence.split()] + [self.tgt_vocab['<eos>']]
            # 对源语言和目标语言的序列进行填充
            src_seq += [self.src_vocab['<pad>']] * \
                (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * \
                (self.tgt_len - len(tgt_seq))
            # 将处理好的序列添加到批次中
            input_batch.append(src_seq)
            output_batch.append([self.tgt_vocab['<sos>']] + ([self.tgt_vocab['<pad>']] *
                                                             (self.tgt_len - 2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])
          # 将批次转换为 LongTensor 类型
        input_batch = torch.LongTensor(input_batch)
        output_batch = torch.LongTensor(output_batch)
        target_batch = torch.LongTensor(target_batch)
        return input_batch, output_batch, target_batch


# # 创建语料库类实例
# corpus = TranslationCorpus(text)
# model = Transformer(corpus)  # 创建模型实例
# criterion = nn.CrossEntropyLoss()  # 损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)  # 优化器
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
# print(f'device: {device}')
# print(f'Training on {device_name}')

# epochs = 5000  # 训练轮次
# for epoch in range(epochs):  # 训练 100 轮
#     optimizer.zero_grad()  # 梯度清零
#     enc_inputs, dec_inputs, target_batch = corpus.make_batch(
#         batch_size)  # 创建训练数据
#     outputs, _, _, _ = model(enc_inputs, dec_inputs)  # 获取模型输出
#     loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)),
#                      target_batch.view(-1))  # 计算损失
#     if (epoch + 1) % 100 == 0:  # 打印损失
#         print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
#     loss.backward()  # 反向传播
#     optimizer.step()  # 更新参数


# # 替换你的整个测试（推理）代码块 (从 307 行开始)

# print("======== 开始翻译测试 ========")
# test_enc_inputs, _, _ = corpus.make_batch(batch_size=1)  # 采样一个句子

# # 初始化解码器输入，以 <sos> token 开始
# # 确保它是 [batch_size, seq_len] = [1, 1] 的形状
# dec_inputs = torch.LongTensor([[corpus.tgt_vocab['<sos>']]])

# preds = []
# max_len = 15  # 设置最大生成长度，防止死循环

# for _ in range(max_len):
#     # 1. 将当前的 [1, L] 输入送入模型
#     dec_out, _, _, _ = model(test_enc_inputs, dec_inputs)

#     # 2. 获取最后一个时间步的 logits [1, L, V] -> [1, V]
#     last_token_logits = dec_out[:, -1, :]

#     # 3. 找到概率最高的 token 索引
#     # .max() 返回 (values, indices)
#     # keepdim=True 保持形状为 [1, 1]，以便于拼接
#     pred_idx = last_token_logits.data.max(1, keepdim=True)[1]

#     next_token_idx = pred_idx.item()

#     # 4. 如果是 <eos>，则停止
#     if corpus.tgt_idx2word[next_token_idx] == '<eos>':
#         break

#     preds.append(next_token_idx)

#     # 5. 自动回归：将预测的 token 拼接到 dec_inputs
#     # [1, L] + [1, 1] -> [1, L+1]
#     dec_inputs = torch.cat([dec_inputs, pred_idx], dim=1)

# # 打印结果
# translated_sentence = [corpus.tgt_idx2word[idx] for idx in preds]
# # 过滤掉 <pad> token，它们不应该出现在源句子中
# input_sentence = ' '.join([corpus.src_idx2word[idx.item(
# )] for idx in test_enc_inputs[0] if idx.item() != corpus.src_vocab['<pad>']])

# print(f"源句: {input_sentence}")
# print(f"译句: {' '.join(translated_sentence)}")
