import torch

from dataset import Events
from model import Model

size=3387

train=Events(0,int(size*0.7))
test=Events(int(size*0.7),size)
model=Model()
#model.load_state_dict(torch.load("model.pt"))

#adam优化器，论文里的学习率是0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_value=[]

for epoch in range(60):

    for idx,event in enumerate(train):
        optimizer.zero_grad()

        loss,result=model(event)
        loss.backward()
        optimizer.step()


        if idx%100==0:
            #print("第{}个sample".format(idx))

            testsize=size-int(size*0.7)

            loss_acc=0

            for i in test:
                testLoss,testResult=model(i)
                loss_acc+=float(testLoss)

            loss_value.append(loss_acc/testsize)
            print(loss_acc/testsize,flush=True)


torch.save(model.state_dict(),"model.pt")#模型保存

# plt.plot([i for i in range(len(loss_value))],loss_value)
# plt.show()


