#评估模块

from dataset import Events
from model import Model
import torch

model=Model()
model.load_state_dict(torch.load("model60.pt"))

size=3387
test=Events(int(size*0.7),size)


correct=0

for idx,x in enumerate(test):

    print("[{}/{}]".format(idx,len(test)))

    loss,result=model((x[0].view(-1,130,300),x[1]))

    if result>=0.5 and x[1]==1 or result<0.5 and x[1]==0:
        correct+=1

#输出准确率
print(correct/len(test))