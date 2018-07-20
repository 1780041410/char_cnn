import re
import pickle
from gensim.models import word2vec
import logging
import  numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
try:
    from langconv import *
except ImportError:
    from langconv import *
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line
def file_cht2chs(file,chs_file):
    file=open(file,'r',encoding='utf-8')
    chs_file=open(chs_file,'a',encoding='utf-8')
    while True:
        line=file.readline()
        if not line:
            break
        line=cht_to_chs(line)
        chs_file.write(line)
def word_replace(line,replace_list,replace_space_list):
    for replace in replace_list:
        line=line.replace(replace[0],replace[1])
    for r_p in replace_space_list:
        line = line.replace(r_p, '')
    line = line.replace(' ', '，')
    while True:
        if line.find('。。')==-1:
            break
        else:
            line = line.replace('。。', '。')
    while True:
        if line.find('，，')==-1:
            break
        else:
            line = line.replace('，，', '，')
    while True:
        if line.find('？？')==-1:
            break
        else:
            line = line.replace('？？', '？')
    while True:
        if line.find(',')==-1:
            break
        else:
            line = line.replace(',', '，')
    line = line.replace('\t', ',')
    return line
def file_word_replace(file,replace_file,is_train=True):
    file = open(file, 'r', encoding='utf-8')
    replace_file = open(replace_file, 'a', encoding='utf-8')
    replace=open('data/replace', 'r', encoding='utf-8').read().split('\n')
    replace_space_list = open('data/replace_space', 'r', encoding='utf-8').read().split('\n')
    replace_list=[re.split('=') for re in replace]
    i=0
    if is_train:
        replace_file.write('question1,question2,label\n')
    else:
        replace_file.write('question1,question2\n')
    while True:
        # print(i)
        i+=1
        line = file.readline()
        if not line:
            break
        line = word_replace(line,replace_list,replace_space_list)
        replace_file.write(line)
def train_csv_file():
    jieba.load_userdict('data/user_defined.txt')
    df=pd.read_csv('data/replace_chs_train.csv').fillna('0')
    question = {}
    i=1
    k=1
    id_list=[]
    qid1_list=[]
    qid2_list=[]
    question1_tokens_list=[]
    question2_tokens_list = []
    question1_pos_list=[]
    question2_pos_list = []
    for q1,q2 in zip(df['question1'],df['question2']):
        id_list.append(k)
        k+=1
        if q1 in question:
            qid1=question[q1]
        else:
            question[q1]=i
            qid1=i
            i+=1
        qid1_list.append(qid1)
        question1_tokens_list.append(' '.join([c.word for c in pseg.cut(q1)]))
        question1_pos_list.append(' '.join([c.flag for c in pseg.cut(q1)]))
        if q2 in question:
            qid2=question[q2]
        else:
            question[q2]=i
            qid2=i
            i+=1
        qid2_list.append(qid2)
        question2_tokens_list.append(' '.join([c.word for c in pseg.cut(q2)]))
        question2_pos_list.append(' '.join([c.flag for c in pseg.cut(q2)]))
    df['id'] = id_list
    df['qid1'] = qid1_list
    df['qid2'] = qid2_list
    df['question1_tokens'] = question1_tokens_list
    df['question2_tokens'] = question2_tokens_list
    df['question1_pos'] = question1_pos_list
    df['question2_pos'] = question2_pos_list
    df.to_csv('data/train.csv',index=False,encoding='utf-8')
def dev_csv_file():
    jieba.load_userdict('data/user_defined.txt')
    df=pd.read_csv('data/replace_chs_dev.csv').fillna('0')
    question = {}
    i=1
    k=1
    id_list=[]
    # qid1_list=[]
    # qid2_list=[]
    question1_tokens_list=[]
    question2_tokens_list = []
    question1_pos_list=[]
    question2_pos_list = []

    for q1,q2 in zip(df['question1'],df['question2']):
        id_list.append(k)
        k+=1
        # if q1 in question:
        #     qid1=question[q1]
        # else:
        #     question[q1]=i
        #     qid1=i
        #     i+=1
        # qid1_list.append(qid1)
        question1_tokens_list.append(' '.join([c.word for c in pseg.cut(q1)]))
        question1_pos_list.append(' '.join([c.flag for c in pseg.cut(q1)]))
        # if q2 in question:
        #     qid2=question[q2]
        # else:
        #     question[q2]=i
        #     qid2=i
        #     i+=1
        # qid2_list.append(qid2)
        question2_tokens_list.append(' '.join([c.word for c in pseg.cut(q2)]))
        question2_pos_list.append(' '.join([c.flag for c in pseg.cut(q2)]))
    df['id'] = id_list
    # df['qid1'] = qid1_list
    # df['qid2'] = qid2_list
    df['question1_tokens'] = question1_tokens_list
    df['question2_tokens'] = question2_tokens_list
    df['question1_pos'] = question1_pos_list
    df['question2_pos'] = question2_pos_list
    df.to_csv('data/dev.csv',index=False,encoding='utf-8')
def pos_model_build():
    pos_set=set()
    df = pd.read_csv('data/dev.csv')
    for pos in df['question1_pos']:
        for p in pos.strip().split():
            pos_set.add(p)
    for pos in df['question2_pos']:
        for p in pos.strip().split():
            pos_set.add(p)
    df = pd.read_csv('data/train.csv')
    for pos in df['question1_pos']:
        for p in pos.strip().split():
            pos_set.add(p)
    for pos in df['question2_pos']:
        for p in pos.strip().split():
            pos_set.add(p)
    data=pd.DataFrame()
    def f(i):
        l=np.zeros(50)
        l[i]=1
        return l
    for i in range(50):
        if i==0:
            data['None']=f(i)
        else:
            data[pos_set.pop()]=f(i)
    data.to_csv('data/pos2vec.csv',encoding='utf-8',index=False)
def wiki_file():
    wikifile=open('data/zh_wiki','a',encoding='utf-8')
    train=pd.read_csv('data/train.csv').fillna('0')
    dev = pd.read_csv('data/dev.csv').fillna('0')
    for i in range(10):
        for q in train['question1_tokens']:
            wikifile.write(q + '\n\n')
        for q in train['question2_tokens']:
            wikifile.write(q + '\n\n')
        for q in dev['question1_tokens']:
            wikifile.write(q + '\n\n')
        for q in dev['question2_tokens']:
            wikifile.write(q.strip() + '\n\n')
def word_file():
    wikifile=open('data/word2vec_file','a',encoding='utf-8')
    train=pd.read_csv('data/train.csv').fillna('0')
    dev = pd.read_csv('data/dev.csv').fillna('0')
    for i in range(10):
        for q in train['question1_tokens']:
            wikifile.write(q + '\n\n')
        for q in train['question2_tokens']:
            wikifile.write(q + '\n\n')
        for q in dev['question1_tokens']:
            wikifile.write(q + '\n\n')
        for q in dev['question2_tokens']:
            wikifile.write(q.strip() + '\n\n')
def word_vec_wiki():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'data/zh_wiki')
    model = word2vec.Word2Vec(sentences, size=400, window=8, min_count=5, workers=4)
    model.save('data/word2vecModel')
def word_vec():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(u'data/word2vec_file')
    model = word2vec.Word2Vec(sentences, size=400, window=8, min_count=5, workers=4)
    model.save('data/word2vec')
def vocab_build():
    df_train = pd.read_csv('data/train.csv')
    df_dev = pd.read_csv('data/dev.csv')
    df = pd.concat([df_train, df_dev])
    sens1=df['question1']
    sens2 = df['question2']
    word2id = {}
    for sent_ in sens1:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    for sent_ in sens2:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<EOS>']=0
    word2id['<PAD>'] = 0
    with open('data/char2id.pkl', 'wb') as fw:
        pickle.dump(word2id, fw)

def word_connection2vec():
    min_count=5
    df_train = pd.read_csv('data/train.csv')
    df_dev = pd.read_csv('data/dev.csv')
    df = pd.concat([df_train, df_dev])
    d=pd.concat([df['question1_tokens'],df['question2_tokens']])
    df=pd.DataFrame()
    df['question_tokens']=d
    word_conn={}
    for question_tokens in df['question_tokens']:
        word_list=question_tokens.split()
        i=0
        while i< len(word_list):
            if i==0:
                if not ('START_tag',word_list[i]) in word_conn:
                    word_conn['START_tag',word_list[i]]=[len(word_conn)+1,1]
                else:
                    word_conn['START_tag', word_list[i]][1]+=1

            elif i==(len(word_list)-1):
                if not (word_list[i-1], word_list[i]) in word_conn:
                    word_conn[word_list[i-1], word_list[i]] = [len(word_conn)+1,1]
                else:
                    word_conn[word_list[i - 1], word_list[i]][1]+=1
                if not (word_list[i],'END_tag') in word_conn:
                    word_conn[ word_list[i],'END_tag']=[len(word_conn)+1,1]
                else:
                    word_conn[word_list[i],'END_tag'][1]+=1
            else:
                if not (word_list[i-1], word_list[i]) in word_conn:
                    word_conn[word_list[i-1], word_list[i]] = [len(word_conn)+1,1]
                else:
                    word_conn[word_list[i - 1], word_list[i]][1]+=1

            i+=1
    low_freq_words = []
    for w_c, [conn_id, conn_freq] in word_conn.items():
        if conn_freq< min_count :
            low_freq_words.append(w_c)
    for w_c in low_freq_words:
        del word_conn[w_c]
    new_id = 1
    for w_c in word_conn.keys():
        word_conn[w_c] = new_id
        new_id += 1
    word_conn['UNK','UNK']=new_id
    word_conn['PAD', 'PAD'] =0
    with open('data/conn2id.pkl', 'wb') as fw:
        pickle.dump(word_conn, fw)
    # print(len(word_conn))
    # for i in word_conn:
    #     print(i)




# file_cht2chs('data/dev.txt','data/chs_dev.txt')
# file_cht2chs('data/train.txt','data/chs_train.txt')
# file_word_replace('data/chs_train.txt','data/replace_chs_train.csv')
# file_word_replace('data/chs_dev.txt','data/replace_chs_dev.csv',is_train=False)
# train_csv_file()
# dev_csv_file()
# pos_model_build()
# wiki_file()
# word_file()
# word_vec()
# vocab_build()
# word_connection2vec()