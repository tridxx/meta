import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import json
import jieba


# 读取数据
# news_label = ['财经', '产品行为', '交往', '竞赛行为', '人生', '司法', '灾害', '组织关系', '组织行为']
# news_label = range(9)
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
                    _D[_i].append(list(jieba.cut(l['text'], cut_all=False)))
                    break
    return _D


'''记得改数据路径'''
f1 = 'C:\\Users\\86150\\Desktop\\data\\train.json'
t = load__data(f1)


# dtype = torch.FloatTensor
# embedding_dim = 3
# n_hidden = 5
# num_classes = 2  # 0 or 1
# sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
# labels = [1, 1, 1, 0, 0, 0]
# word_list = " ".join(sentences).split()
# word_list = list(set(word_list))
# word_dict = {w: i for i, w in enumerate(word_list)}
# vocab_size = len(word_dict)
#
# inputs = []
# for sen in sentences:
#     inputs.append(np.asarray([word_dict[n] for n in sen.split()]))
# inputs = np.array(inputs)
# targets = []
# for out in labels:
#     targets.append(out)
# input_batch = Variable(torch.LongTensor(inputs))
# target_batch = Variable(torch.LongTensor(targets))
# query_batch_label = 1
# query_batch = 1
# 创建出36个二分类任务
# 这个地方的word_list这边要重新修改一下，把里面用jieba切分的部分扔掉，得到input。
class TaskData:
    def __init__(self, txt1, cls1, txt2, cls2):
        self.cls1 = cls1
        self.cls2 = cls2
        _text = txt1 + txt2
        self.labels = np.ones(len(txt1) + len(txt2))
        for _i in range(len(txt1)):
            self.labels[_i] = 0  # 将txt的标签记为0
        word_list = []
        for _i in range(len(_text)):
            word_list = word_list + _text[_i]
        word_list = list(set(word_list))
        self.word_dict = {w: i for i, w in enumerate(word_list)}
        self.vocab_size = len(self.word_dict)
        self.input = []
        for sen in range(len(_text)):
            self.input.append(np.asarray([self.word_dict[n] for n in _text[sen]]))


TaskDataMatrix = list(36, dtype=TaskData)
num = 0
for i in range(9):
    for j in range(9):
        TaskDataMatrix[num] = TaskData(t[i], i, t[j], j)


# 超参数和数据集的接口都要补上去
# 使用BiLSTM_Attention作为分类器
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


model = BiLSTM_Attention()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoches = 20
tasks = 35  # C_9^2 -1
lr_meta = 0.0001
batch = 3

theta_matrix = torch.zeros(size=[tasks, 1])
theta_matrix = theta_matrix.float()
meta_phi = torch.randn(size=[1, 1])
meta_phi = meta_phi.float()
meta_grd = torch.zeros_like(meta_phi)


# training
def train(epoch):
    global meta_grd, meta_phi
    optimizer.zero_grad()
    output, attention = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

    loss_sum = 0.0
    for i in range(tasks):
        model.W.data = meta_phi.data
        optimizer.zero_grad()
        output, attention = model(input_batch)
        loss = criterion(output, target_batch)
        loss_sum = loss_sum + loss.data.item()
        loss.backward()
        optimizer.step()
        theta_matrix[1, :] = model.W
    for i in range(tasks):
        model.W.data = theta_matrix[i]
        optimizer.zero_grad()
        output, attention = model(query_batch)
        loss = criterion(output, query_batch_label)
        loss.backward()
        optimizer.step()
        meta_grd = meta_grd + model.W
    meta_phi = meta_phi - lr_meta * meta_grd / tasks
    if epoch + 1 % 200 == 0:
        print("the Epoch is {:04d}".format(epoch), "the Loss is {:.4f}".format(loss_sum / tasks))


for epoch in range(epoches):
    train(epoch)
