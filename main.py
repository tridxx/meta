import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

dtype = torch.FloatTensor
embedding_dim = 3
n_hidden = 5
num_classes = 2  # 0 or 1
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word_dict)

inputs = []
for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))
inputs = np.array(inputs)
# print(inputs)
targets = []
for out in labels:
    targets.append(out)
input_batch = Variable(torch.LongTensor(inputs))
target_batch = Variable(torch.LongTensor(targets))


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

# Training
for epoch in range(5000):
    optimizer.zero_grad()
    output, attention = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Test
test_text = 'sorry hate you'
tests = np.array([np.asarray([word_dict[n] for n in test_text.split()])])
test_batch = Variable(torch.LongTensor(tests))
# Predict
predict, _ = model(test_batch)
predict = predict.data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text, "is Bad Mean...")
else:
    print(test_text, "is Good Mean!!")
