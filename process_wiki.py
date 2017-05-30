import logging
import os.path
import sys
import re
import jieba
import pandas
import numpy as np
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

#process the xml.bz2 file
program=os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
inp = 'zhwiki-latest-pages-articles.xml.bz2'
outp = 'wiki.zh.text'
space = " "
i=0

output=open(outp,'w')
wiki = WikiCorpus(inp,lemmatize=False,dictionary={})
for text in wiki.get_texts():
    #print(text)
    output.write(space.join(str(v) for v in text)+"\n")
    i=i+1
    if(i%10000==0):
        logger.info("saved"+str(i)+" articles")

output.close()
logger.info("finished")

#process with the encoding
with open('wiki.zh.text','rb') as fin,open('wiki.zh.text.fan','wb')as fout:
    line1 = fin.readline()
    while line1:
        s = eval(line1.strip())
        fout.write(s.encode('utf-8')
        line1 = fin.readlin()

# > opencc -i wiki.zh.text.fan -o wiki.zh.text.jian -c zht2zhs.ini
#convert tradition characters into simplified characters

#filter some strange characters 
with open('wiki.zh.text.jian', 'rb') as fin, open('wiki.zh.text.you', 'wb') as fout:
    line1 = fin.readline()
    while line1:
        temp = re.sub(r"[^\u4e00-\u9fa5]",'',line1.encode('utf-8').decode('utf-8'))
        fout.write(temp.encode('utf-8'))
        fout.write(b'\n')
        line1 = fin.readline()

#segment the words using jieba
with open('wiki.zh.text.you','r') as fin, open('wiki.zh.text.seg','wb') as fout:
    line = fin.readline()
    while line:
        a = jieba.cut(line)
        fout.write(' '.join(a).encode('utf-8'))
        line = fin.readline()

#train and save the model
outpp = 'wiki.model'
inpp = 'wiki.zh.text.seg'
model = word2vec.Word2Vec(LineSentence(inpp),size=300,min_count=5)
model.save(outpp)
