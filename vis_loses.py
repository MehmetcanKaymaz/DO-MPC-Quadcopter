import matplotlib.pyplot as plt
import numpy as np


batch_size=[8,16,32,64]
learning_rate=[0.1,0.01,0.001,0.0001]
N=100

class datas:
    def __init__(self):
        self.data_train=np.zeros(N)
        self.data_val=np.zeros(N)

    def setter(self,train,test):
        self.data_train=train
        self.data_val=test

data_list=[]

for i in range(16):
    data_train=np.loadtxt("Loss/TrainLoss_vel_{}.txt".format(i))
    data_val=np.loadtxt("Loss/ValidLoss_vel_{}.txt".format(i))
    data=datas()
    data.setter(train=data_train,test=data_val)
    data_list.append(data)

epp=np.linspace(0,N+1,N+1)

fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i,j].plot(epp,data_list[i*4+j].data_train,epp,data_list[i*4+j].data_val)
        axs[i,j].legend(["Train","Validation"])
        axs[i,j].set_title("Batch Size:{}  Learning Rate:{}".format(batch_size[i],learning_rate[j]))
        axs[i,j].set(ylabel='Loss')


plt.show()

