import jieba
import pickle
import re
import numpy as np
from Parameters import Parameters as pm
import tensorflow as tf
from seq2seq import Seq2Seq

with open('./data/word2id.pkl', 'rb') as fr:
    word_dict = pickle.load(fr)


def get_data(filename):
    fp = open(filename, 'r', encoding='utf-8')
    x_data = []
    for line in fp:
        x_data.append(list(jieba.lcut(line)))
    return x_data

def label2id(filename):
    '''
    将文字按照字典转数字
    '''
    x_label = get_data(filename)
    x_id = []
    length = len(x_label)
    for i in range(length):
        w = []
        for key in x_label[i]:
            if key not in word_dict:
                key = '<UNK>'
            w.append(word_dict[key])
        x_id.append(w)
    return x_id

def evaluate():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/Seq2Seq')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    content = label2id(pm.eva_data)
    pre_ids = model.predict(session, content)
    return pre_ids


if __name__ == '__main__':
    pm = pm
    model = Seq2Seq()
    label_ids = evaluate()
    print(label_ids)
    with open(pm.eva_data, 'r', encoding='utf-8') as f:
        sentences = [line.strip('\n') for line in f]

    new_dict = {v: k for k, v in word_dict.items()} #将原字典key与value值转换
    n = 0
    for label in label_ids:
        print(sentences[n])
        label = label.T
        for i in range(pm.beam_size):
            predict_list = label[i]
            sentence = []
            for idx in predict_list:
                sentence.append(new_dict[idx])
            print(re.sub('<EOS>', '', ''.join(sentence)))
        n += 1
