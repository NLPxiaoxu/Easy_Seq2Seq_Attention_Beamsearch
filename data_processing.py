# -*- coding:utf-8 -*-
import re
import jieba
import pickle
import numpy as np
from collections import Counter
from Parameters import Parameters as pm
import tensorflow.contrib.keras as kr

word_dict = {'<PAD>': 0, '<UNK>': 1, '<GO>': 2, '<EOS>': 3}

def get_data(filename):
    '''
    文本处理，获得训练数据，建立字典
    '''
    fp = open(filename, 'r', encoding='utf-8')
    j = 4
    x_data, y_data, alldata, wordlist, x_label, y_label = [], [], [], [], [], []
    for line in fp:
        line = line.strip('\n')
        line = re.sub('/', '', line)
        alldata.append(line)

    for i in range(len(alldata)):
        if alldata[i] == 'E' and i < len(alldata)-5:
            n = alldata.index('E', i+1)
            if n - i > 3:
                line1 = re.sub('M', '', alldata[i+1])
                line2 = re.sub('M', '', alldata[i+2])
                line3 = re.sub('M', '', alldata[i+3])
                sentence = line1.strip(' ')+','+line2.strip(' ')
                sentence = re.sub(' ', '', sentence)
                line3 = re.sub(' ', '', line3)
                x_data.append(''.join(sentence))
                y_data.append(line3.strip(' '))
            elif 2 < n - i <= 3:
                line1 = re.sub('M', '', alldata[i + 1])
                line2 = re.sub('M', '', alldata[i + 2])
                line1 = re.sub(' ', '', line1)
                line2 = re.sub(' ', '', line2)
                x_data.append(line1.strip(' '))
                y_data.append(line2.strip(' '))

    for i in range(len(x_data)):
        X = list(jieba.lcut(x_data[i]))
        Y = list(jieba.lcut(y_data[i]))
        if len(Y) > 1 and len(X) >1:
            wordlist.extend(X)
            wordlist.extend(Y)
            x_label.append(X)
            y_label.append(Y)

    counter = Counter(wordlist)
    counter_pari = counter.most_common(19996)
    word, _ = list(zip(*counter_pari))
    for key in word:
        word_dict[key] = j
        j += 1

    with open('./data/word2id.pkl', 'wb') as fw:  # 将建立的字典 保存
        pickle.dump(word_dict, fw)
    return x_label, y_label

#x_label, y_label = get_data(pm.train_data)
#print(x_label[:10])
#print(y_label[:10])
#print(word_dict)

def label2id(filename):
    '''
    将文字按照字典转数字，并在y_id结尾加上<EOS>
    '''
    x_label, y_label = get_data(filename)
    with open('./data/word2id.pkl', 'rb') as fr:
        word_dict = pickle.load(fr)
    x_id, y_id = [], []
    length = len(x_label)
    for i in range(length):
        w = []
        for key in x_label[i]:
            if key not in word_dict:
                key = '<UNK>'
            w.append(word_dict[key])
        x_id.append(w)

    for k in range(length):
        h = []
        for word in y_label[k]:
            if word not in word_dict:
                word = '<UNK>'
            h.append(word_dict[word])
        h.append(word_dict['<EOS>'])
        y_id.append(h)

    return x_id, y_id
#x_id, y_id = label2id(pm.train_data)
#print(x_id[:10])
#print(y_id[:10])

def batch_iter(x, y, batch_size = pm.batch_size):
    x = np.array(x)
    y = np.array(y)
    length = len(x)
    indices = np.random.permutation(length)
    num_batch = int((length-1)/batch_size) + 1
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start = i * batch_size
        end = min((i+1) * batch_size, length)
        yield x_shuffle[start: end], y_shuffle[start: end]

def process_seq(x_batch):
    '''
    :param x_batch: 计算一个batch里面最长句子 长度n
    :param y_batch:动态RNN 保持同一个batch里句子长度一致即可，sequence为实际句子长度
    :return: 对所有句子进行padding,长度为n
    '''
    seq_len = []
    max_len = max(map(lambda x: len(x), x_batch))  # 计算一个batch中最长长度
    for i in range(len(x_batch)):
        seq_len.append(len(x_batch[i]))

    x_pad = kr.preprocessing.sequence.pad_sequences(x_batch, max_len, padding='post', truncating='post')

    return x_pad, seq_len

def process_target(y_batch):
    seq_len = []
    max_len = max(map(lambda y: len(y), y_batch))  # 计算一个batch中最长长度
    for i in range(len(y_batch)):
        seq_len.append(len(y_batch[i]))

    y_pad = kr.preprocessing.sequence.pad_sequences(y_batch, max_len, padding='post', truncating='post')

    return y_pad, seq_len
