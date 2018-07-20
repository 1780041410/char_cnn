import numpy as np
import pandas as pd
l0=[]
for i in range(10):
    l0.append(pd.read_csv('pre'+str(i)+'.csv')['pre0'].values.tolist())
l1=[]
for i in range(10):
    l1.append(pd.read_csv('pre'+str(i)+'.csv')['pre1'].values.tolist())
l0=np.array(l0)
l1=np.array(l1)
rel=[]
for l00 ,l11 in zip(l0,l1):
    r=[]
    for l000,l111 in zip(l00,l11):
        if l000>l111:
            r.append(0)
        else:
            r.append(1)
    rel.append(r)

re1=[]
for i in range(10000):
    k=0
    for j in range(10):
        k+=rel[j][i]
    re1.append(k)
print(len(re1))
label=pd.read_csv('1.csv')['result']
l0_sum=np.sum(l0,axis=0)
print(len(l0_sum))
l1_sum=np.sum(l1,axis=0)
re=[]
for r ,l in zip(re1,label):
    if r>5:
        re.append(1)
    elif r<5:
        re.append(0)
    else:
        re.append(l)
for i in re:
    print(i)
df = pd.DataFrame()
df['test_id'] = [i for i in range(10000)]
df['result'] = re
df.to_csv('result.csv', encoding='utf-8', index=False)
cnn_feature_dev = pd.DataFrame()