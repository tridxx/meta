# 开发者：李丞正旭
# 开发时间： 2022/11/13 18:11
'''提取文本'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import json
import jieba


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append(l['text'])
    return D


'''提取种类'''


def load__data(filename):
    classify = ['财经/交易', '产品行为', '交往', '竞赛行为', '人生', '司法行为', '灾害/意外', '组织行为', '组织关系']
    _D = []
    for _i in range(9):
        _D.append([])
    num = 0
    with open(filename, encoding='utf-8') as f:
        for l in f:
            num += 1
            l = json.loads(l)
            classify_name = l['event_list'][0]['class']

            for _i, c in enumerate(classify):
                if c in classify_name:
                    _x = list(jieba.cut(l['text'], cut_all=False))
                    _x1 = []
                    for _j in range(len(_x)):
                        if _x[_j] in {'，', '。', ',', '.', '?', '？', '!', '！', ' ', '：', ':', '"', '”', '“', '<', '>',
                                      '《', '》'}:
                            continue
                        else:
                            _x1.append(_x[_j])
                    _D[_i].append(_x1)
                    break
    return _D


class TaskData:
    def __init__(self, txt1, cls1, txt2, cls2):
        self.cls1 = cls1
        self.cls2 = cls2
        _text = txt1 + txt2
        max_sen_len = 0
        self.labels = np.ones(len(txt1) + len(txt2))
        for _i in range(len(txt1)):
            self.labels[_i] = 0  # 将txt的标签记为0
        word_list = []
        for _i in range(len(_text)):
            word_list = word_list + _text[_i]
            max_sen_len = max(max_sen_len, len(_text[_i]))
        word_list = list(set(word_list))
        max_sen_len = 500  # 改了一下
        self.word_dict = {w: i for i, w in enumerate(word_list)}
        self.vocab_size = len(self.word_dict)
        self.input = np.zeros((len(_text), max_sen_len), dtype=int)
        for sen in range(len(_text)):
            tmp = np.array([self.word_dict[n] for n in _text[sen]])
            for _i in range(len(_text[sen])):
                self.input[sen][_i] = tmp[_i]
        # 下面划分出训练集和测试集
        testset_len1 = int(len(txt1) / 4)
        testset_len2 = int(len(txt2) / 4)
        self.input1 = np.zeros((len(_text) - testset_len1 - testset_len2, max_sen_len), dtype=int)
        self.label1 = np.zeros(len(_text) - testset_len2 - testset_len1, dtype=int)
        for _i in range(len(txt1) - testset_len1):
            self.input1[_i] = self.input[_i]
            self.label1[_i] = 0
        for _i in range(len(txt1), len(_text) - testset_len2):
            self.input1[_i - testset_len1] = self.input[_i]
            self.label1[_i - testset_len1] = 1
        self.testset = np.zeros((testset_len2 + testset_len1, max_sen_len), dtype=int)
        self.testlabel = np.zeros(testset_len1 + testset_len2, dtype=int)
        for _i in range(testset_len1):
            self.testset[_i] = self.input[len(txt1) - 1 - _i]
            self.testlabel[_i] = 0
        for _i in range(testset_len2):
            self.testset[_i] = self.input[len(_text) - 1 - _i]
            self.testlabel[_i + testset_len1] = 1


'''记得改数据路径'''
f1 = 'C:\\Users\\86150\\Desktop\\data\\train.json'
text = load__data(f1)
cls = ['财经/交易', '产品行为', '交往', '竞赛行为', '人生', '司法行为', '灾害/意外', '组织行为', '组织关系']
x = TaskData(text[0], cls[0], text[1], cls[1])

dtype = torch.FloatTensor
embedding_dim = 3
n_hidden = 8
num_classes = 2
vocab_size = x.vocab_size


class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention


class RNN_ATTs(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0, hidden_size2=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out


# model = RNN_ATTs(vocab_size, embedding_dim, n_hidden, 2)
model = BiLSTM_Attention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
input_batch = Variable(torch.LongTensor(x.input1))
target_batch = Variable(torch.LongTensor(x.label1))
for epoch in range(100):
    optimizer.zero_grad()
    output, _ = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
# test
test_batch = Variable(torch.LongTensor(x.testset))
# predict = model(test_batch)
predict, _ = model(test_batch)
predict = predict.data.max(1, keepdim=True)[1]
correct_num = 0
for i in range(len(x.testlabel)):
    if predict[i][0] == x.testlabel[i]:
        correct_num = correct_num + 1
print("test in same domain: ", correct_num / len(x.testlabel))

y1 = TaskData(text[2], cls[2], text[3], cls[3])
y2 = TaskData(text[4], cls[4], text[5], cls[5])

test_batch = Variable(torch.LongTensor(y1.testset))
predict, _ = model(test_batch)
predict = predict.data.max(1, keepdim=True)[1]
correct_num = 0
for i in range(len(y1.testlabel)):
    if predict[i][0] == y1.testlabel[i]:
        correct_num = correct_num + 1
print("test in y1 domain: ", correct_num / len(y1.testlabel))


test_batch = Variable(torch.LongTensor(y2.testset))
predict, _ = model(test_batch)
predict = predict.data.max(1, keepdim=True)[1]
correct_num = 0
for i in range(len(y2.testlabel)):
    if predict[i][0] == y2.testlabel[i]:
        correct_num = correct_num + 1
print("test in y2 domain: ", correct_num / len(y2.testlabel))
