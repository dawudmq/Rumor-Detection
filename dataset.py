from torch.utils.data import Dataset
import numpy as np
import torch

#继承pytorch的数据集类
class Events(Dataset):

    def __init__(self,start,end):
        super(Events,self).__init__()
        self.file=np.load("cutWord-dataSet.npy")[start:end]
        self.embeddingMartix=np.load("embeddingMartix.npy").item()
        self.pad='<pad>'#填充符

        self.maxLenth=130#所有句子的最大长度

        for idx in range(self.file.shape[0]):

            #找到数据集句子的最大长度
            # if len(self.file[idx][0])>self.maxLenth:
            #     self.maxLenth=len(self.file[idx][0])

            #将所有句子填充到最大长度
            for j in range(self.maxLenth-len(self.file[idx][0])):
                self.file[idx][0].append(self.pad)


    #idx为下标，获取数据集中第idx个元素
    def __getitem__(self, idx):

        text,label=self.file[idx][0],self.file[idx][1]
        textList=[]

        #从embeddingMartix中找相应的词向量，找不到就用全零向量代替
        for i in text:
            if i in self.embeddingMartix:
                textList.append(self.embeddingMartix[i])
            else:
                textList.append(np.zeros((300)))

        return torch.tensor(textList,dtype=torch.float32),torch.tensor(int(label))

    def __len__(self):
        return self.file.shape[0]

