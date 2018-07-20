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
config=config.Config()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()
logger.info("Loading data...")
filter_sizes = "3,4,5"
config.filter_sizes=list(map(int, filter_sizes.split(",")))
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        logger.info("Loading rnn model")
        print('Loading rnn model')
        rnn = model6(config)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './runs/20180618135204/checkpoints/model-18000')
        df_train = pd.read_csv(config.devfile)
        v2id = read_dictionary(config.vocab_path)
        pos_vec = pd.read_csv('data/pos2vec.csv')
        model = word2vec.Word2Vec.load(config.vec_model)
        print('word2vec')
        def dev_step(config):
            config.is_train=False
            batches = batch_yield_dev( config,df_train,v2id,pos_vec,model)
            prelist = []
            score_list=[]
            config.lstm_dropout_keep_prob = 1.0
            config.char_dropout_keep_prob = 1.0
            config.feature_dropout_keep_prob = 1.0
            config.mlp_dropout_keep_prob = 1.0
            for batch in batches:
                char_batch, char_real, word_batch, word_real, feature_batch= zip(batch)
                c=char_batch[0]
                w=word_batch[0]
                c_real=char_real[0]
                w_real=word_real[0]
                f_batch=feature_batch[0]
                # print(k)
                # k+=1
                feed_dict = {
                    rnn.input_char: c,
                    rnn.input_word: w,
                    rnn.batch_size: len(c),
                    rnn.real_len: c_real,
                    rnn.input_feature: f_batch,
                    rnn.real_len_word: w_real,
                }
                predictions ,score= sess.run(
                    [ rnn.predictions,rnn.scores],
                    feed_dict)
                prelist.append(predictions)
                score_list.append(score)
            return prelist,score_list
        prelist,score_list = dev_step( config)
        df=pd.DataFrame()
        df['test_id'] = [i for i in range(10000)]
        re=[]
        for pre in prelist:
            for p in pre:
                re.append(p)
        # print(re)
        df['result']=re
        df.to_csv('result.csv',encoding='utf-8',index=False)

