import pandas as pd
from numpy import *
import tensorflow as tf
import numpy as np
import os
import time
import config
import datetime
import logging
from data_utils import *
from rnn_model import *
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
import pandas as pd
class Config1():
    def __init__(self,):
        self.trainfile = '.\\data\\train.csv'
        self.devfile = '.\\data\\dev.csv'
        self.vocab_path = '.\\data\\char2id.pkl'
        self.optimizer='Adam'
        self.lstm_dropout_keep_prob=1.0
        self.word_dropout_keep_prob=1.0
        self.char_dropout_keep_prob = 1.0
        self.feature_dropout_keep_prob = 1.0
        self.mlp_dropout_keep_prob = 1.0
        self.char_num = 1501
        self.feature_size = 32
        self.num_classes = 2
        self.char_sequence_length=202
        self.word_sequence_length=133
        self.char_embedding_dim=300
        self.word_embedding_size = 450
        self.char_lstm_size=300
        self.word_lstm_size = 450
        self.char_attention_size=300
        self.word_attention_size = 300
        self.feature_mlp_size = 100
        self.MLP_size=300
        self.dev_percentage = 0.1
        self.rate = 1e-3
        self.batch_size=150
        self.batch_size_dev = 100
        self.layer_num = 2
        self.is_train=True
        self.shuffle = True
        self.num_epochs = 100
        self.num_checkpoints = 20
        self.checkpoint = ''
        self.evaluate_every = 500
        self.checkpoint_every = 2000

config=config.Config()
def prediction(m):
    re = []
    error=0
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            print('Loading rnn model')
            rnn = model_dev1(config)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './runs/20180526121958/checkpoints/model-14000')
            df_train = pd.read_csv(config.trainfile)
            v2id = read_dictionary(config.vocab_path)
            pos_vec = pd.read_csv('data/pos2vec.csv')
            model = word2vec.Word2Vec.load('data/word2vecModel')

            def dev_step(config):
                config.is_train = False
                batches = batch_yield(config, df_train, v2id, pos_vec, model,90000)
                loss_sum = 0
                accuracy_sum = 0
                count = 0
                recall_sum = 0
                prelist=[]
                config.lstm_dropout_keep_prob = 1.0
                config.word_dropout_keep_prob = 1.0
                config.char_dropout_keep_prob = 1.0
                config.feature_dropout_keep_prob = 1.0
                config.mlp_dropout_keep_prob = 1.0
                for batch in batches:
                    char_batch, char_real, word_batch, word_real, feature_batch, label = zip(batch)
                    c = char_batch[0]
                    w = word_batch[0]
                    c_real = char_real[0]
                    w_real = word_real[0]
                    f_batch = feature_batch[0]
                    l = label[0]
                    feed_dict = {
                        rnn.input_char: c,
                        rnn.input_word: w,
                        rnn.batch_size: len(char_batch),
                        rnn.real_len: c_real,
                        rnn.input_feature: f_batch,
                        rnn.real_len_word: w_real,
                        rnn.input_label: l
                    }
                    loss, accuracy, recall ,predictions= sess.run(
                        [ rnn.loss, rnn.accuracy, rnn.recall,rnn.predictions],
                        feed_dict)
                    loss_sum = loss_sum + loss
                    accuracy_sum = accuracy_sum + accuracy
                    recall_sum = recall_sum + recall
                    count = count + 1
                    prelist.append(predictions)
                print(loss_sum, accuracy_sum, recall_sum)

                loss = loss_sum / count
                error = 1-(accuracy_sum / count)
                recall = recall_sum / count
                return prelist,error
            prelist,error = dev_step(config)
            re = []
            # print(prelist)
            for pre in prelist:
                for p in pre:
                    re.append(p)
    return  m,re ,error
m,re,er=prediction(m=10)
print(len(re))
print(er)



def adaboost():
    num_lte=100
    test_num = 10000
    best={}
    a=[1,1,1,1,1,1,1,1,1,1]
    weakClassArr=[]
    d = mat(ones((test_num, 1)) / test_num)
    for i in range(num_lte):
        m,classEst,error=prediction(10)
        alpha=float(0.5*log((1.0-error*a[i])/max(error*a[i],0.11111111111)))

        weakClassArr.append(best)

