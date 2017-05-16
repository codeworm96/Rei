import pandas as pd
import numpy as np
import jieba

from WordMap import WordMap

neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None)
pn=pd.concat([pos,neg],ignore_index=True)

cw = lambda x: list(jieba.cut(x, HMM=False))
pn['words'] = pn[0].apply(cw)

w = []
for i in pn['words']:
  w.extend(i)

wordmap = WordMap()
wordmap.build(w, 35000)
wordmap.save("word.map")
