import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np


torch.manual_seed(1)    # reproducible

print("Loading datas ...")
matrix = np.loadtxt("Datas/all_data.txt", dtype=np.float32)  #veriyi float32 cinsinden matrix degisenine aktariyor.
train = matrix[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
train = torch.from_numpy(train)
label = matrix[:,[13,14,15,16]]
label = torch.from_numpy(label)

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(13, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.LeakyReLU(),
	torch.nn.Linear(512, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512, 4),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 256
EPOCH = 10000

torch_dataset = Data.TensorDataset(train, label)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)


epp_loss=[]

print("Training ....")

for epoch in range(EPOCH):
    if epoch%500==0:
        torch.save(net.state_dict(), "Models/checkpoint{}.pth".format(epoch))
        print("checkpoint{} saved!".format(epoch))

    loss_arr=[]
    print("Epoch {} started...".format(epoch))
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x
        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
        loss_arr.append(loss.item())
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    epp_loss.append(np.mean(np.array(loss_arr)))

np.savetxt("Loss.txt",np.array(epp_loss))
