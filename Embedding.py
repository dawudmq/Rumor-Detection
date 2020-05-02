#构建embedding矩阵，使用第一段注释代码获得所有词汇并保存，然后从词向量文件中找出词汇表里每一个词的词向量

import pandas as pd
import jieba
import numpy as np

# file=pd.read_csv("original-rumor-final.csv",encoding="gbk")
# texts=file["text"].tolist()
#
# voc=[]
#
# for text in texts:
#     seg=jieba.cut(text)
#     for word in seg:
#         voc.append(word)
#
#
# voc=set(voc)

voc=np.load("voc.npy").tolist()

embeddingMartix={}#词向量字典，键为词语，值为词向量
word2vec=open("Tencent_AILab_ChineseEmbedding.txt",encoding="utf-8").readlines()

print("词向量文件加载完毕...")

for vec in word2vec:

    if len(embeddingMartix)%100==0:
        print("[{}/{}]".format(len(embeddingMartix),len(voc)))

    vector=vec.strip().split(" ")
    if vector[0] in voc:
        embeddingMartix[vector[0]]=[float(dimension) for dimension in vector[1:]]

np.save("embeddingMartix.npy",embeddingMartix)