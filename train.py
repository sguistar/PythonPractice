import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import utils.d2l as d2l
import os
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入Transformer模型
from model.Transformer import Transformer

# 下载并加载英法数据集
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入"英语－法语"数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()
    
def preprocess_nmt(text):
    """预处理"英语－法语"数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化"英语－法语"数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

# 定义适用于英法翻译的语料库类
class NMTCorpus:
    def __init__(self, raw_text, num_examples=1000):
        # 预处理文本
        text = preprocess_nmt(raw_text)
        # 词元化
        self.source, self.target = tokenize_nmt(text, num_examples)
        
        # 计算源语言和目标语言的最大句子长度
        self.src_len = max(len(seq) for seq in self.source) + 1  # +1 for <eos>
        self.tgt_len = max(len(seq) for seq in self.target) + 2  # +2 for <sos> and <eos>
        
        # 创建源语言和目标语言的词汇表
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        
        # 创建索引到单词的映射
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}
        
    def create_vocabularies(self):
        # 统计源语言和目标语言的单词频率
        src_counter = Counter(word for seq in self.source for word in seq)
        tgt_counter = Counter(word for seq in self.target for word in seq)
        
        # 创建源语言和目标语言的词汇表，并为每个单词分配一个唯一的索引
        # 保留前10000个最常见的词
        src_vocab_list = [word for word, _ in src_counter.most_common(10000)]
        tgt_vocab_list = [word for word, _ in tgt_counter.most_common(10000)]
        
        src_vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        src_vocab.update({word: i+3 for i, word in enumerate(src_vocab_list)})
        
        tgt_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        tgt_vocab.update({word: i+4 for i, word in enumerate(tgt_vocab_list)})
        
        return src_vocab, tgt_vocab
    
    def encode(self, tokens, vocab, max_len, is_source=True):
        """将词元序列编码为索引序列"""
        if is_source:
            # 源语言序列：添加 <eos> 标记
            tokens = tokens[:max_len-1]  # 为 <eos> 留出空间
            token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens] + [vocab['<eos>']]
        else:
            # 目标语言序列：添加 <sos> 和 <eos> 标记
            tokens = tokens[:max_len-2]  # 为 <sos> 和 <eos> 留出空间
            token_ids = [vocab['<sos>']] + [vocab.get(token, vocab['<unk>']) for token in tokens] + [vocab['<eos>']]
        
        # 填充序列
        token_ids += [vocab['<pad>']] * (max_len - len(token_ids))
        return token_ids
    
    def make_batch(self, batch_size, test_batch=False):
        """创建批次数据"""
        input_batch, output_batch, target_batch = [], [], []
        
        # 随机选择句子索引
        indices = torch.randperm(len(self.source))[:batch_size]
        
        for idx in indices:
            src_tokens = self.source[idx]
            tgt_tokens = self.target[idx]
            
            # 编码源语言和目标语言序列
            src_seq = self.encode(src_tokens, self.src_vocab, self.src_len, is_source=True)
            tgt_seq = self.encode(tgt_tokens, self.tgt_vocab, self.tgt_len, is_source=False)
            
            # 添加到批次中
            input_batch.append(src_seq)
            if test_batch:
                # 测试时只需要 <sos> 标记开始
                output_batch.append([self.tgt_vocab['<sos>']] + [self.tgt_vocab['<pad>']] * (self.tgt_len - 1))
            else:
                # 训练时使用目标序列的前 n-1 个词作为输入
                output_batch.append(tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])  # 目标序列从第二个词开始
        
        # 转换为张量
        input_batch = torch.LongTensor(input_batch)
        output_batch = torch.LongTensor(output_batch)
        target_batch = torch.LongTensor(target_batch)
        
        return input_batch, output_batch, target_batch

# 下载并处理数据
print("正在下载数据集...")
raw_text = read_data_nmt()
print("正在处理数据集...")
corpus = NMTCorpus(raw_text, num_examples=10000)  # 使用前10000个样本

print(f"源语言词汇表大小: {len(corpus.src_vocab)}")
print(f"目标语言词汇表大小: {len(corpus.tgt_vocab)}")
print(f"源语言最大长度: {corpus.src_len}")
print(f"目标语言最大长度: {corpus.tgt_len}")

# 创建模型实例
print("正在创建模型...")
model = Transformer(corpus)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=corpus.tgt_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 训练参数
epochs = 100
batch_size = 64

print("开始训练...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # 计算每个epoch的批次数量
    num_batches = len(corpus.source) // batch_size
    
    for batch_idx in range(num_batches):
        optimizer.zero_grad()
        
        # 创建训练数据批次
        enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size)
        
        # 将数据移到设备上
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        target_batch = target_batch.to(device)
        
        # 前向传播
        outputs, _, _, _ = model(enc_inputs, dec_inputs)
        
        # 计算损失
        loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 每10个批次打印一次信息
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}')
    
    # 每个epoch打印平均损失
    avg_loss = total_loss / num_batches
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
    
    # 每10个epoch进行一次翻译测试
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            # 选择一个测试样本
            test_enc_inputs, _, _ = corpus.make_batch(1, test_batch=True)
            test_enc_inputs = test_enc_inputs.to(device)
            
            # 初始化解码器输入
            dec_inputs = torch.LongTensor([[corpus.tgt_vocab['<sos>']]]).to(device)
            
            # 逐步生成翻译
            for _ in range(corpus.tgt_len - 1):
                dec_out, _, _, _ = model(test_enc_inputs, dec_inputs)
                # 获取最后一个时间步的预测
                pred = dec_out[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # 如果预测是结束标记，则停止
                if pred.item() == corpus.tgt_vocab['<eos>']:
                    break
                    
                # 将预测结果添加到解码器输入中
                dec_inputs = torch.cat([dec_inputs, pred], dim=1)
            
            # 将结果转换为文本
            input_sentence = ' '.join([corpus.src_idx2word.get(idx, '<unk>') 
                                     for idx in test_enc_inputs[0].cpu().numpy() 
                                     if idx not in [corpus.src_vocab['<pad>'], corpus.src_vocab['<eos>']]])
            
            output_sentence = ' '.join([corpus.tgt_idx2word.get(idx, '<unk>') 
                                      for idx in dec_inputs[0].cpu().numpy()[1:]  # 跳过<sos>
                                      if idx not in [corpus.tgt_vocab['<pad>'], corpus.tgt_vocab['<eos>'], corpus.tgt_vocab['<sos>']]])
            
            print(f"源句: {input_sentence}")
            print(f"译句: {output_sentence}")
            print("-" * 50)

print("训练完成!")
torch.save(model.state_dict(), '../output/transformer_model.pth')