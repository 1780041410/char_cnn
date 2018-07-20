import config
import sys, pickle, os, random
import numpy as np
import jieba
from gensim.models import word2vec
import pandas as pd
def label2id(label,config):
    l=[]
    for i in label:
        if int(i) ==1:
            if config.num_classes==2:
                l.append([0,1])
            else:
                l.append([1])
        else:
            if config.num_classes == 2:
                l.append([1,0])
            else:
                l.append([0])
    return l
def get_input_feature(is_train=True):
    feature = []
    if is_train:
        nlp_feature_train=pd.read_csv('data/train_feature.csv')
        non_nlp_feature_train = pd.read_csv('data/non_nlp_features_train.csv')
        for x in range(len(nlp_feature_train['num_repeat_char'])):
            feature.append(list(non_nlp_feature_train.ix[x]) + list(nlp_feature_train.ix[x]))
    else:
        nlp_feature_dev = pd.read_csv('data/dev_feature.csv')
        non_nlp_feature_dev = pd.read_csv('data/non_nlp_features_dev.csv')
        for x in range(len(nlp_feature_dev['num_repeat_char'])):
            feature.append(list(non_nlp_feature_dev.ix[x]) + list(nlp_feature_dev.ix[x]))
    return feature
def get_input_feature_dev():
    feature = []

    nlp_feature_dev = pd.read_csv('data/dev_feature.csv')
    non_nlp_feature_dev = pd.read_csv('data/non_nlp_features_dev.csv')
    for x in range(len(nlp_feature_dev['num_repeat_char'])):
        feature.append(list(non_nlp_feature_dev.ix[x]) + list(nlp_feature_dev.ix[x]))
    return feature
def get_input_feature1(is_train=True):
    feature = []
    if is_train:
        nlp_feature_train=pd.read_csv('data/nlp_features_train.csv')
        for x in range(len(nlp_feature_train['cwc_min'])):
            feature.append( list(nlp_feature_train.ix[x]))
    else:
        nlp_feature_dev = pd.read_csv('data/nlp_features_test.csv')
        for x in range(len(nlp_feature_dev['cwc_min'])):
            feature.append( list(nlp_feature_dev.ix[x]))
    return feature
def get_input_word(sens1,sens2,pos1,pos2,pos_vec,model,max_sen_length):

    pad=np.zeros(model.vector_size+pos_vec['None'].size)
    sens_vec=[]
    sens1_vec=[]
    sens2_vec = []
    real=[]
    for sen1 ,p1 in zip(sens1,pos1):
        sen1_vec = []
        word_list=sen1.split()
        w_v=[]
        p_v=[]
        for word in word_list:
            try:
                w_v.append(model[word])
            except:
                w_v.append(model['None'])
        for p in p1:
            try:
                p_v.append(pos_vec[p])
            except:
                p_v.append( pos_vec['None'])
        sens1_vec.append([list(w_)+list(p_) for w_,p_ in zip(w_v,p_v)])
        # sens1_vec.append(sen1_vec)
    for sen2 ,p2 in zip(sens2,pos2):
        sen2_vec = []
        word_list=sen2.split()
        w_v = []
        p_v = []
        for word in word_list:
            try:
                w_v.append(model[word])
            except:
                w_v.append(model['None'])
        for p in p2:
            try:
                p_v.append(pos_vec[p])
            except:
                p_v.append(pos_vec['None'])
        sens2_vec.append([list(w_)+list(p_) for w_,p_ in zip(w_v,p_v)])
        # sens2_vec.append(sen2_vec)
    for i in range(len(sens1_vec)):
        sens1_vec[i].append(pad)
        sens_vec.append(sens1_vec[i] + sens2_vec[i])
    input_word = []
    seq_len_list = []
    for seq in sens_vec:
        seq_length = len(seq)
        seq_=[]
        for i in range(max_sen_length):
            if i<seq_length:
                seq_.append(seq[i])
            else:
                seq_.append(pad)
        #seq_ = []#list.copy(seq).expad * max(max_sen_length - len(seq), 0)
        input_word.append(seq_)
        seq_len_list.append(min(seq_length, max_sen_length))
    return np.array(input_word), np.array(seq_len_list)
def get_input_char(sens1,sens2, word2id,max_sen_length):
    sens1_id=[]
    sens1_len=[]
    for sent in sens1:
        sentence_id = []
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word = '<UNK>'
            sentence_id.append(word2id[word])

        # sentence_id = sentence_id[:] + [0] * max(max_sen_length - len(sent), 0)
        sens1_id.append(sentence_id)
        sens1_len.append(len(sent))

    sens2_id = []
    sens2_len = []
    for sent in sens2:
        sentence_id = []
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word = '<UNK>'
            sentence_id.append(word2id[word])
        # sentence_id = sentence_id[:] + [0] * max(max_sen_length - len(sent), 0)
        sens2_id.append(sentence_id)
        sens2_len.append(len(sent))
    EOS = word2id['<EOS>']
    char_ = []
    real_length=[]
    for i in range(len(sens1_id)):
        sens1_id[i].append(EOS)
        char_.append(sens1_id[i] + sens2_id[i])
        real_length.append(sens1_len[i]+sens2_len[i]+1)
    char_list=[]
    seq_len_list=[]
    for seq in char_:
        seq = list(seq)
        seq_ = seq[:] + [0] * max(max_sen_length - len(seq), 0)
        char_list.append(seq_)
        seq_len_list.append(min(len(seq), max_sen_length))
    return np.array(char_list),np.array(seq_len_list)
def read_dictionary(vocab_path):

    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id
def read_conn_dictionary(conn_path):

    vocab_path = os.path.join(conn_path)
    with open(vocab_path, 'rb') as fr:
        conn2id = pickle.load(fr)
    print('conn_size:', len(conn2id))
    return conn2id

def max_sen_length(sens1,sens2):
    sens = [s1 + s2 for s1, s2 in zip(sens1, sens2)]
    max_sen_length=max([len(x) for x in sens])+1
    return max_sen_length
def word_max_sen_length(sens1,sens2):
    sens1_list=[x.split() for x in sens1]
    sens2_list =[x.split() for x in sens2]
    sen_length=[]
    for s1,s2 in zip(sens1_list,sens2_list):
        sen_length.append(len(s1)+len(s2)+1)
    max_sen_length=max(sen_length)
    return max_sen_length
def batch_yield(config,df_train,v2id,pos_vec,model,dev_sample_index):


    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    feature = np.array(get_input_feature()[:dev_sample_index],dtype=float)
    question1=np.array(list(df_train['question1'])[:dev_sample_index])
    question2 = np.array(list(df_train['question2'])[:dev_sample_index])
    question1_tokens = np.array(list(df_train['question1_tokens'])[:dev_sample_index])
    question2_tokens = np.array(list(df_train['question2_tokens'])[:dev_sample_index])
    question1_pos = np.array(list(df_train['question1_pos'])[:dev_sample_index])
    question2_pos = np.array(list(df_train['question2_pos'])[:dev_sample_index])
    label = np.array(label2id(df_train['label'],config)[:dev_sample_index])
    feature_dev = np.array(get_input_feature()[dev_sample_index:],dtype=float)
    question1_dev  = np.array(list(df_train['question1'])[dev_sample_index:])
    question2_dev  = np.array(list(df_train['question2'])[dev_sample_index:])
    question1_tokens_dev  = np.array(list(df_train['question1_tokens'])[dev_sample_index:])
    question2_tokens_dev  = np.array(list(df_train['question2_tokens'])[dev_sample_index:])
    question1_pos_dev  = np.array(list(df_train['question1_pos'])[dev_sample_index:])
    question2_pos_dev  = np.array(list(df_train['question2_pos'])[dev_sample_index:])
    label_dev  = np.array(label2id(df_train['label'],config)[dev_sample_index:])
    if config.is_train:
        data_size = 90000
        batch_size = config.batch_size
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        for epoch in range(config.num_epochs):
            if config.shuffle:
                np.random.seed(epoch)
                shuffle_indices = np.random.permutation(np.arange(len(feature)))
                q1_shuffled = question1[shuffle_indices]
                q2_shuffled = question2[shuffle_indices]
                q1_t_shuffled = question1_tokens[shuffle_indices]
                q2_t_shuffled = question2_tokens[shuffle_indices]
                q1_p_shuffled = question1_pos[shuffle_indices]
                q2_p_shuffled = question2_pos[shuffle_indices]
                feature_shuffled = feature[shuffle_indices]
                label_shuffled = label[shuffle_indices]
            else:
                q1_shuffled = question1
                q2_shuffled = question2
                q1_t_shuffled = question1_tokens
                q2_t_shuffled = question2_tokens
                q1_p_shuffled = question1_pos
                q2_p_shuffled = question2_pos
                feature_shuffled = feature
                label_shuffled = label
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                input_char, char_real = get_input_char(q1_shuffled[start_index:end_index],
                                                       q2_shuffled[start_index:end_index], v2id,
                                                       config.char_sequence_length)
                input_word, word_real = get_input_word(q1_t_shuffled[start_index:end_index],
                                                       q2_t_shuffled[start_index:end_index],
                                                       q1_p_shuffled[start_index:end_index],
                                                       q2_p_shuffled[start_index:end_index], pos_vec, model,
                                                       config.word_sequence_length)
                # print(input_word.shape)
                yield input_char, char_real, input_word, word_real, feature_shuffled[
                                                                    start_index:end_index], label_shuffled[
                                                                                            start_index:end_index]
    else:
        data_size = 10000
        batch_size = config.batch_size_dev
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            input_char, char_real = get_input_char(question1_dev [start_index:end_index],
                                                   question2_dev[start_index:end_index], v2id,
                                                   config.char_sequence_length)
            input_word, word_real = get_input_word(question1_tokens_dev[start_index:end_index],
                                                   question2_tokens_dev[start_index:end_index],
                                                   question1_pos_dev[start_index:end_index],
                                                   question2_pos_dev[start_index:end_index], pos_vec, model,
                                                   config.word_sequence_length)
            # print(input_word.shape)
            yield input_char, char_real, input_word, word_real, feature_dev[
                                                                start_index:end_index], label_dev[
                                                                                        start_index:end_index]

def batch_yield_dev(config,df_train,v2id,pos_vec,model):


    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    feature = np.array(get_input_feature_dev(),dtype=float)
    question1=np.array(list(df_train['question1']))
    question2 = np.array(list(df_train['question2']))
    question1_tokens = np.array(list(df_train['question1_tokens']))
    question2_tokens = np.array(list(df_train['question2_tokens']))
    question1_pos = np.array(list(df_train['question1_pos']))
    question2_pos = np.array(list(df_train['question2_pos']))
    data_size = 10000
    print(data_size)
    batch_size = config.batch_size_dev
    print(batch_size)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        input_char, char_real = get_input_char(question1[start_index:end_index],
                                               question2[start_index:end_index], v2id,
                                               config.char_sequence_length)
        input_word, word_real = get_input_word(question1_tokens[start_index:end_index],
                                               question2_tokens[start_index:end_index],
                                               question1_pos[start_index:end_index],
                                               question2_pos[start_index:end_index], pos_vec, model,
                                               config.word_sequence_length)
        # print(input_word.shape)
        yield input_char, char_real, input_word, word_real, feature[
                                                            start_index:end_index]

