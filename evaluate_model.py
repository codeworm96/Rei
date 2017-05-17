import pandas as pd
import numpy as np
import jieba
import pickle

from keras.preprocessing import sequence

from WordMap import WordMap
from KerasLSTM import KerasLSTM

wordmap = WordMap()
wordmap.load("word.map")

model = KerasLSTM()
'''
neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None)

pos['mark']=1
neg['mark']=0
pn=pd.concat([pos,neg],ignore_index=True)

cw = lambda x: list(jieba.cut(x, HMM=False))
pn['words'] = pn[0].apply(cw)

get_sent = lambda x: [wordmap.lookup(w) for w in x]
pn['sent'] = pn['words'].apply(get_sent)
'''
pn = pickle.load(open('./data','rb'))
maxlen = 100
print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

k = 5
splits = np.split(pn.sample(frac=1), k)
fp = 0
fn = 0
tp = 0
tn = 0

for i in range(0, k):
    print('Round', i)
    validate = splits[i]
    chosen = []
    for j in range(0, k):
        if i != j:
            chosen.append(splits[j])
    train = pd.concat(chosen)

    xt = np.array(list(train['sent']))
    yt = np.array(list(train['mark']))

    model.build(wordmap.size(), maxlen, xt, yt)

    res = model.predict(np.array(list(validate['sent'])))
    mark = np.array(list(validate['mark']))
    for i in range(0, len(validate)):
        if res[i] >= 0.5:
            if mark[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if mark[i] == 1:
                fn += 1
            else:
                tn += 1

print("{}-fold cross validation:".format(k))
print("false positive:", fp)
print("true positive:", tp)
print("false negative:", fn)
print("true negative:", tn)
print("accuracy:", (tn + tp) / (fp + tp + fn + tn))
print("precision:", tp / (tp + fp))
print("recall:", tp / (tp + fn))
