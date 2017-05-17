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

neg=pd.read_excel('simplied_data/neg.xls',header=None,index=None)
pos=pd.read_excel('simplied_data/pos.xls',header=None,index=None)

pos['mark']=1
neg['mark']=0
pn=pd.concat([pos,neg],ignore_index=True)

cw = lambda x: list(jieba.cut(x, HMM=False))
pn['words'] = pn[0].apply(cw)

get_sent = lambda x: [wordmap.lookup(w) for w in x]
pn['sent'] = pn['words'].apply(get_sent)
#print(pn['sent'])
#print(pn.shape)
#print(len(pn['sent'][0]))
maxlen = 100
print("Pad sequences (samples x time)")
pickle.dump(pn,open('./data','wb'))

#pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
#print(pn['sent'])
#xa = np.array(list(pn['sent']))
#ya = np.array(list(pn['mark']))
#print(xa)
#model.build(wordmap.size(), maxlen, xa, ya)
#model.save('model.h5')
