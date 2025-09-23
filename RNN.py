import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

import random


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers,
                          dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


def init_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.RNN):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


# Sample data preparation
batch_size = 2
corpus = ["我 喜欢 玩具", "我 爱 祖国", "我 讨厌 挨打"]
vocab_list = list(set(" ".join(corpus).split()))  # 创建词汇表
vocab_size = len(vocab_list)
word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}
idx_to_word = {idx: word for idx, word in enumerate(vocab_list)}


def make_data():
    input_data = []  # 定义输入批处理列表
    target_data = []  # 定义目标（标签）批处理列表
    sentences = random.sample(corpus, batch_size)  # 随机采样句子
    # print(sentences)
    for sent in sentences:
        words = sent.split()  # 用空格将句子分隔成多个词
        # 将除最后一个词以外的所有词的索引作为输入，形成一个list
        input_i = [word_to_idx[i] for i in words[:-1]]
        # 将最后一个词的索引作为目标（标签）
        target = word_to_idx[words[-1]]  # 创建目标（标签）数据
        input_data.append(input_i)  # 将输入添加到输入批处理列表
        target_data.append(target)  # 将目标（标签）添加到目标（标签）批处理列表
    input_data = torch.LongTensor(input_data)  # 将输入数据转换为张量
    target_data = torch.LongTensor(target_data)  # 将目标（标签）数据转换为张量
    return input_data, target_data  # 返回输入批处理和目标（标签）批处理数据


# Example usage
model = RNN(input_size=vocab_size, hidden_size=10,
            output_size=vocab_size, n_layers=2, dropout=0.5)
print(model)
model.apply(init_weights)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(4500):
    optimizer.zero_grad()
    input_data, target_data = make_data()
    # Initialize hidden state for each batch
    hidden = model.init_hidden(batch_size=input_data.size(0))
    output, hidden = model(input_data, hidden)
    loss = loss_fn(output, target_data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

# Testing
print("\n--- Testing ---")
input_strs = [['我', '讨厌'], ['我', '喜欢'],['我', '爱']]  # input_strs中每个元素是一个句子
input_indices = [[word_to_idx[word] for word in seq]
                 for seq in input_strs]  # 将句子转换为索引
input_batch = torch.LongTensor(input_indices)  # 转换为张量

# Re-initialize hidden state for prediction
test_hidden = model.init_hidden(batch_size=input_batch.size(0))
predict_output, _ = model(input_batch, test_hidden)
predict = predict_output.data.max(1, keepdim=True)[1]

predict_strs = [idx_to_word[n.item()] for n in predict.squeeze()]
for input_seq, pred in zip(input_strs, predict_strs):
    print(input_seq, '->', pred)
