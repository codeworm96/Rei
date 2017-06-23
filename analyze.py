import pandas as pd
import numpy as np
import jieba
import math
import gensim

from keras.preprocessing import sequence

from WordMap import WordMap
from KerasLSTM import KerasLSTM
from tfModel import tfModel

def analyze(data, model_select):
    result=[]
    if model_select==0:
        wordmap = WordMap()
        wordmap.load("word.map")
        model = KerasLSTM()
        model.load("model.h5")
        toTrain=[]
        lengths=[]
        for weibos in data:
            lengths.append(len(weibos))
            for weibo in weibos:
                words = [wordmap.lookup(w) for w in jieba.cut(weibo, HMM=False)]
                toTrain.append(words)
        
        padded = sequence.pad_sequences(toTrain, maxlen=100)
        
        padded = np.array(padded)
        #print(padded.shape)
        emotion = model.predict(padded)
        #print(emotion)
        emo_index=0
        #print("length:")
        #print(lengths)
        for index,item in enumerate(lengths):
            if item==1:
                result.append(emotion[emo_index][0])
                emo_index+=1
            else:
                temp = emotion[emo_index][0]*0.3 + emotion[emo_index+1][0]*0.7
                for i in range(2,item):
                    temp = temp*0.3+emotion[emo_index+i][0]*0.7
                result.append(temp)
                emo_index+=item
                
    elif model_select==1:
        model = tfModel()
        wordModel = gensim.models.Word2Vec.load('wiki.model')
        padding = [0 for x in range(0,300)]

        max_seqlen=100
        seqlen=[]
        toTrain=[]
        lengths=[]
        for weibos in data:
            lengths.append(len(weibos))
            for weibo in weibos:
                words = list(jieba.cut(weibo,HMM=False))
                print(words)
                x=[]
                seqlen.append(len(words))
                if len(words)<max_seqlen:
                    for j in words:
                        try:
                            x.append(wordModel[j])
                        except:
                            x.append(padding)
                    for i in range(max_seqlen-len(words)):
                        x.append(padding)
                else:
                    for i in range(max_seqlen):
                        try:
                            x.append(wordModel[words[i]])
                        except:
                            x.append(padding)
                toTrain.append(x)
        
        toTrain=np.array(toTrain)
        emotion=model.predict(max_seqlen,seqlen,toTrain)
        emo_index=0
        for index,item in enumerate(lengths):
            if item==1:
                result.append(emotion[emo_index][1])
                emo_index+=1
            else:
                temp = emotion[emo_index][1]*0.3 + emotion[emo_index+1][1]*0.7
                for i in range(2,item):
                    temp = temp*0.3+emotion[emo_index+i][1]*0.7
                result.append(temp)
                #print(result)
                emo_index+=item
        #print(result)    
    return result


#data = [["五月了，很开心","真丧啊",'哈哈哈哈哈哈哈'],['晚上要去吃辣了'],['哈哈哈哈哈哈哈哈']]
#result = analyze(data,0)
#print(result)

