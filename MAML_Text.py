import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# noinspection PyPep8Naming
import torch.nn.functional as F
import json
import jieba
import random as rd


# 读取数据
# news_label = ['财经', '产品行为', '交往', '竞赛行为', '人生', '司法', '灾害', '组织关系', '组织行为']
# news_label = range(9)
def load__data(filename):
    classify = ['财经/交易', '产品行为', '交往', '竞赛行为', '人生', '司法行为', '灾害/意外', '组织行为', '组织关系']
    _D = []
    for _i in range(9):
        _D.append([])
    _num = 0
    with open(filename, encoding='utf-8') as f:
        for l in f:
            _num += 1
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


TaskDataMatrix = []
num = 0
for i in range(9):
    for j in range(i + 1, 9):
        TaskDataMatrix.append(TaskData(t[i], i, t[j], j))

dtype = torch.FloatTensor
embedding_dim = 3
n_hidden = 7
num_classes = 2
vocab_size = max(TaskDataMatrix[i].vocab_size for i in range(36))
print()


# 超参数和数据集的接口都要补上去
# 使用BiLSTM_Attention作为分类器
class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    @staticmethod
    def attention_net(lstm_output, final_state):
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
epoches = 2
tasks = 30  # C_9^2-6
lr_meta = 0.0001
batch = 3

theta_matrix = torch.zeros(size=[tasks, 1])
theta_matrix = theta_matrix.float()
# params = [p for p in model.parameters() if p.requires_grad]
# meta_phi = torch.randn(size=[1, 1])
# meta_phi = meta_phi.float()


meta_grd = torch.zeros_like(meta_phi)


# training
def train(_epoch):
    global meta_grd, meta_phi
    # optimizer.zero_grad()
    # output, attention = model(input_batch)
    # loss = criterion(output, target_batch)
    # if (_epoch + 1) % 1000 == 0:
    #     print('Epoch:', '%04d' % (_epoch + 1), 'cost =', '{:.6f}'.format(loss))
    # loss.backward()
    # optimizer.step()

    loss_sum = 0.0
    for _i in range(tasks):
        model.W.data = meta_phi.data
        optimizer.zero_grad()
        output, attention = model(TaskDataMatrix[_i].input1)
        loss = criterion(output, TaskDataMatrix[_i].label1)
        loss_sum = loss_sum + loss.data.item()
        loss.backward()
        optimizer.step()
        theta_matrix[1, :] = model.W
    for _i in range(tasks):
        model.W.data = theta_matrix[_i]
        optimizer.zero_grad()
        output, attention = model(TaskDataMatrix[_i].testset)
        loss = criterion(output, TaskDataMatrix[_i].testlabel)
        loss.backward()
        optimizer.step()
        meta_grd = meta_grd + model.W
    meta_phi = meta_phi - lr_meta * meta_grd / tasks
    if _epoch + 1 % 200 == 0:
        print("the Epoch is {:04d}".format(_epoch), "the Loss is {:.4f}".format(loss_sum / tasks))


for epoch in range(epoches):
    train(epoch)

# test
# correct_num = 0
# correct_rate = np.zeros(6)
# for i in range(30, 36):
#     predict, _ = model(TaskDataMatrix[i].input)
#     predict = predict.data.max(1, keepdim=True)[1]
#     correct_num = 0
#     for j in range(len(TaskDataMatrix[i].labels)):
#         if predict[j][0] == TaskDataMatrix[i].labels[j]:
#             correct_num = correct_num + 1
#     correct_rate[i] = correct_num / len(TaskDataMatrix[i].input)
# print(correct_rate)
