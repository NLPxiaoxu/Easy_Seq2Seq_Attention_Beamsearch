# -*- coding:utf-8 -*-
class Parameters(object):

    num_epochs = 9
    num_layers = 2
    embedding_size = 128
    batch_size = 128
    beam_size = 3
    hidden_dim = 128
    vocab_size = 20000

    learning_rate = 0.005
    clip = 5.0
    lr = 0.8
    keep_pro = 0.5


    train_data = './data/dgk_shooter_min.conv'
    eva_data = './data/evaluate.txt'