# 开发者：李丞正旭
# 开发时间： 2022/11/13 18:11
'''提取文本'''
import json
import jieba
import numpy as np


# def load_data(filename):
#     D = []
#     with open(filename, encoding='utf-8') as f:
#         for l in f:
#             l = json.loads(l)
#             D.append(l['text'])
#     return D


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


'''记得改数据路径'''
f1 = 'C:\\Users\\86150\\Desktop\\data\\train.json'
text = load__data(f1)
cls = ['财经/交易', '产品行为', '交往', '竞赛行为', '人生', '司法行为', '灾害/意外', '组织行为', '组织关系']
x = TaskData(text[0], cls[0], text[1], cls[1])


# print(len(res))

# 产生class


# class TaskData:
#     def __init__(self, txt1, cls1, txt2, cls2):
#         self.cls1 = cls1  # 将cls1映射为label=0
#         self.cls2 = cls2  # 将cls2映射为label=1
#         text = txt1 + txt2
#         self.labels = np.ones(len(txt1) + len(txt2))
#         for _i in range(len(txt1)):
#             self.labels[_i] = 0
#         word_list = []  # 要删掉
#         temp_list = []
#         for _i in range(len(text)):
#             _t = list(jieba.cut(text[_i], cut_all=False))  # 要改掉
#             word_list = word_list + _t
#             temp_list.append(_t)
#         word_list = list(set(word_list))
#         self.word_dict = {w: i for i, w in enumerate(word_list)}
#         self.vocab_size = len(self.word_dict)
#         self.input = []
#         for sen in range(len(text)):
#             self.input.append(np.asarray([self.word_dict[n] for n in temp_list[sen]]))
