import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
df_train = pd.read_csv('data/train.csv')
df_dev = pd.read_csv('data/dev.csv')
df=pd.concat([df_train, df_dev])
print(df)
pos_vec=pd.read_csv('data/pos2vec.csv')
# df1=pd.read_csv('C:/Users/panch/Desktop/8049.csv')['result']
df2=pd.read_csv('C:/Users/panch/Desktop/805.csv')['result']
df3=pd.read_csv('C:/Users/panch/Desktop/80529.csv')['result']
df4=pd.read_csv('C:/Users/panch/Desktop/8053.csv')['result']
df5=pd.read_csv('C:/Users/panch/Desktop/8060.csv')['result']
df6=pd.read_csv('C:/Users/panch/Desktop/8061.csv')['result']
df7=pd.read_csv('C:/Users/panch/Desktop/8065.csv')['result']
df8=pd.read_csv('C:/Users/panch/Desktop/808.csv')['result']
df9=pd.read_csv('C:/Users/panch/Desktop/8083.csv')['result']
df10=pd.read_csv('C:/Users/panch/Desktop/811.csv')['result']
df11=pd.read_csv('C:/Users/panch/Desktop/8113.csv')['result']
df12=pd.read_csv('C:/Users/panch/Desktop/812.csv')['result']

re=[]
for i in range(10000):
    a=+0.04*df2[i]+0.05*df3[i]+0.05*df4[i]+0.065*df5[i]+0.07*df6[i]+0.08*df7[i]+0.10*df8[i]+0.11*df9[i]+0.14*df10[i]+0.145*df11[i]+0.15*df12[i]
    # a = 0.091 * df1[i] + 0.092 * df2[i] + 0.093 * df3[i] + 0.094 * df4[i] + 0.099 * df5[i] + 0.101 * df6[i] + 0.1063 * df7[
    #     i] + 0.1067 * df8[i] + 0.108 * df9[i]+0.109* df10[i]
    if a>0.5:
        re.append(1)
    else:
        re.append(0)
df=pd.DataFrame()
df['test_id'] = [i for i in range(10000)]
df['result']=re
df.to_csv('result.csv',encoding='utf-8',index=False)
# # with open('data/conn2id.pkl', 'rb') as fr:
# #     word2id = pickle.load(fr)
# #     if ('PAD','PAD') in word2id:
# #         print(1)
#
# #     # print(word2id['PAD','PAD'])
# sess=tf.Session()
# # a=np.array([1,0,0,1,1,1,0,1,1])
# # aa=np.array([0,0,0,0,0,0,0,0,0])
# # b=np.array([0.6,0.6,0.3,0.7,0.9,0.4,0.5,0.9,0.8])
# # ta=tf.Variable(a,dtype=tf.float32,name='a')
# # taa=tf.Variable(aa,dtype=tf.float32,name='a')
# # tb=tf.Variable(b,dtype=tf.float32)
# # sess.run(tf.global_variables_initializer())
# # s=tf.cast(tf.equal(ta,taa),dtype=tf.float32)
# # mul=tf.add(tf.multiply(ta,tb),s)
# # log=tf.log(mul)
# # adda=tf.reduce_sum(log)
# # print(sess.run(log))
# # print(sess.run(tf.div(adda,tf.reduce_sum(ta))))
# tf.constant()