import collections
import json

import jieba
import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# 读取数据
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
        self.word_dict = {w: _j for _j, w in enumerate(word_list)}
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

    def forward(self, x):
        input = self.embedding(x)
        input = input.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1 * 2, len(x), n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, len(x), n_hidden))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention

    def functional_forward(self, x, _params):
        x = F.embedding(x, weight=_params[f'embedding.weight'])
        x = x.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1 * 2, len(x), n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, len(x), n_hidden))
        lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True, weight_ih_l=_params[f'lstm.weight_ih_l0'],
                       weight_hh_l=_params[f'lstm.weight_hh_l0'], bias_ih_l=_params[f'lstm.bias_ih_l0'],
                       bias_hh_l=_params[f'lstm.bias_hh_l0'], bias_ih_l0_reverse=_params[f'lstm.weight_ih_l0_reverse'],
                       bias_hh_l_reverse=_params[f'lstm.bias_hh_l0_reverse'])
        x, (final_hidden_state, final_cell_state) = lstm(x, (hidden_state, cell_state))
        x = x.permute(1, 0, 2)
        attn_output, attention = self.attention_net(x, final_hidden_state)
        x = F.linear(n_hidden * 2, num_classes, bias=_params[f'out.bias'])
        return x, attention


def maml_train(_model, innerlr, supportset, supportlabels, queryset, querylabels, _inner_step, _optimizer,
               is_train=True):
    meta_loss = []
    meta_acc = []
    for supportset, supportlabels, queryset, querylabels in zip(supportset, supportlabels, queryset,
                                                                querylabels):

        fast_weights = collections.OrderedDict(model.named_parameters())
        for _ in range(_inner_step):
            support_logit = model.functional_forward(supportset, fast_weights)
            support_loss = nn.CrossEntropyLoss().cuda()(support_logit, supportlabels)
            grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - innerlr * grads)
                                                   for ((name, param), grads) in zip(fast_weights.items(), grads))

        query_logit = model.functional_forward(queryset, fast_weights)
        query_prediction = torch.max(query_logit, dim=1)[1]

        query_loss = nn.CrossEntropyLoss().cuda()(query_logit, querylabels)
        query_acc = torch.eq(querylabels, query_prediction).sum() / len(querylabels)

        meta_loss.append(query_loss)
        meta_acc.append(query_acc.data.cpu().numpy())

    _optimizer.zero_grad()
    meta_loss = torch.stack(meta_loss).mean()
    meta_acc = np.mean(meta_acc)

    if is_train:
        meta_loss.backward()
        _optimizer.step()

    return meta_loss, meta_acc


if __name__ == '__main__':

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    f1 = 'C:\\Users\\86150\\Desktop\\data\\train.json'
    t = load__data(f1)
    TaskDataMatrix = []
    num = 0
    for i in range(9):
        for j in range(i + 1, 9):
            TaskDataMatrix.append(TaskData(t[i], i, t[j], j))

    vocab_size = max(TaskDataMatrix[i].vocab_size for i in range(36))
    embedding_dim = 3
    num_classes = 2
    n_hidden = 8
    outer_lr = 0.001  # 这是元学习的学习率
    inner_lr = 0.0005
    model = BiLSTM_Attention()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, outer_lr)
    best_acc = 0

    model.train()

    epochs = 10

    tasks = 30

    params = collections.OrderedDict(model.named_parameters())

    for epoch in range(epochs):
        train_acc = []
        val_acc = []
        train_loss = []
        val_loss = []
        for i in range(tasks):
            support_set = TaskDataMatrix[i].input1
            support_labels = TaskDataMatrix[i].label1
            query_set = TaskDataMatrix[i].testset
            query_labels = TaskDataMatrix[i].testlabel
            inner_step = 1
            loss, acc = maml_train(model, inner_lr, support_set, support_labels, query_set, query_labels,
                                   inner_step, optimizer, is_train=True)
            train_loss.append(loss.item())
            train_acc.append(acc)

        for i in range(tasks, 36):
            support_set = TaskDataMatrix[i].input1
            support_labels = TaskDataMatrix[i].label1
            query_set = TaskDataMatrix[i].testset
            query_labels = TaskDataMatrix[i].testlabel
            inner_step = 3
            loss, acc = maml_train(model, inner_lr, support_set, support_labels, query_set, query_labels,
                                   inner_step, optimizer, is_train=False)
            train_loss.append(loss.item())
            train_acc.append(acc)

        print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".
              format(np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)))

        if np.mean(val_acc) > best_acc:
            best_acc = np.mean(val_acc)
            torch.save(model, 'best.pt')
