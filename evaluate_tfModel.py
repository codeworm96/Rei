import pandas as pd
import numpy as np
import jieba
import pickle
import gensim
from tfModel import tfModel

neg=pd.read_excel('whole_data/neg.xls',header=None,index=None)
pos=pd.read_excel('whole_data/pos.xls',header=None,index=None)

pos['mark']=1
neg['mark']=0
pn=pd.concat([pos,neg],ignore_index=True)

cw = lambda x: list(jieba.cut(x, HMM=False))
pn['sent'] = pn[0].apply(cw)

wordModel = gensim.models.Word2Vec.load('wiki.model')

padding = [0 for x in range(0,300)]

max_seqlen=100
seqlen=[]
for line in pn['sent']:
    #print(len(line))
    if len(line)>max_seqlen:
        seqlen.append(100)
    else:
        seqlen.append(len(line))

xt = []
for line in pn['sent']:
    x = []
    if len(line)<max_seqlen:
        #for i in range(max_seqlen-len(line)):
        for j in line:
            try:
                x.append(wordModel[j])
            except:
                x.append(padding)
        for i in range(max_seqlen-len(line)):
            x.append(padding)
    else:
        for i in range(max_seqlen):
            try:
                x.append(wordModel[line[i]])
            except:
                x.append(padding)
    xt.append(x)

#print("qileguaile")
xt = np.array(xt)
#print(data.shape)
rows = xt.shape[0]
#print(rows)
arr = np.arange(rows)
np.random.shuffle(arr)
xt = xt[arr]
#print(xt.shape)
yt = np.array(pn['mark'])
yt = yt[arr]
#print(yt.shape)
seqlen = np.array(seqlen)
seqlen = seqlen[arr]
#print(seqlen.shape)
fp = 0
tp = 0
fn = 0
tn = 0
print("Building model")
model = tfModel()
size = int(rows*4/5)
model.build(max_seqlen, seqlen[:size], xt[:size,:,:], yt[:size], seqlen[size:], xt[size:,:,:], yt[size:])
#print("Start Testing")
#model.test(max_seqlen, seqlen[size:], xt[size:, :, :], yt[size:])

