import pandas as pd
import numpy as np
import jieba

from keras.preprocessing import sequence

from WordMap import WordMap
from KerasLSTM import KerasLSTM

wordmap = WordMap()
wordmap.load("word.map")

model = KerasLSTM()
model.load("model.h5")

while True:
    weibo = input("Input a weibo: ")
    words = [wordmap.lookup(w) for w in jieba.cut(weibo, HMM=False)]
    padded = list(sequence.pad_sequences([words], maxlen=100))
    x = np.array(list(padded))
    emotion = model.predict(x)
    print(emotion)
