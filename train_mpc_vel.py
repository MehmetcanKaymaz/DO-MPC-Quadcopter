import torch
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import argparse
#System Check
#print(torch.cuda.is_available)
parser = argparse.ArgumentParser(description='MPC')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size')
parser.add_argument('--learning_rate', default=0.0001 , type=float,
                    help='learning rate')
parser.add_argument('--index', default=1000 , type=int,
                    help='learning rate')
args = parser.parse_args()



#print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#print (os.getcwd())


#--------------------------Initialization------------------------
#Model daha karışık olacak 2 veya 3 layer eklenecek,batchnorm, pool,dropout,
#desired loss tanımlanacak,
#learning decay


#batch size daha küçük 8,16,32
#learning rate 0.1,0.01,0.001,0.0001



#epoch 1000,2000,3000

# Data Parameters
batch_size = args.batch_size

# Model Parameters
input_size = 9
output_size = 4 


# Train Parameters
learningRate = args.learning_rate
epochs = 1000

index=args.index

#---------------------------DataLoader---------------------------
torch.manual_seed(1)    # reproducible

#print("Loading datas ...")
matrix = np.loadtxt("Dagger-All-Data/Dagger_D0_D1_D2_s.txt", dtype=np.float32)  #veriyi float32 cinsinden matrix degisenine aktariyor.

matrix = torch.from_numpy(matrix)
train_size = int(.9 * len(matrix))
train = matrix[0:train_size,:]
train_input =train[:,[0,1,2,3,4,5,6,7,8]]
train_label = train[:,[9,10,11,12]]
train_size  = len(train)

valid = matrix[train_size:-1,:]
valid_input = valid[:,[0,1,2,3,4,5,6,7,8]]
valid_label = valid[:,[9,10,11,12]]
valid_size  = len(valid)

#------------------------------Model-----------------------------
net = torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, output_size),
    )

net.load_state_dict(torch.load("Models/checkpoint_vel_D1_1000_1000.pth"))
net.to(device)


loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay= 0.001)

#--------------------Train Model--------------------------------
# Make model ready to train
net.train()



train_dataset = Data.TensorDataset(train_input, train_label )
train_loader = Data.DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True, num_workers=2,)

"""test_dataset = Data.TensorDataset(test_input, test_label)
test_loader = Data.DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size, 
    shuffle=True, num_workers=2,)"""

valid_dataset = Data.TensorDataset(valid_input, valid_label)
valid_loader = Data.DataLoader(
    dataset=valid_dataset, 
    batch_size=batch_size, 
    shuffle=True, num_workers=2,)


epp_train_loss=[]
epp_valid_loss=[]
#print("Training ....")
for epoch in range(epochs+1):
    if epoch%100==0:
        torch.save(net.state_dict(), "Models/checkpoint_vel_D2_{}_{}.pth".format(index,epoch))
        #print("checkpoint_{}_{} saved!".format(epoch))


    batch_loss=[]
    for step, (batch_x, batch_y) in enumerate(train_loader): 
        b_x = Variable(batch_x)
        
        b_y = Variable(batch_y)

        prediction = net(b_x.to(device))     # input x and predict based on x
        loss = loss_func(prediction, b_y.to(device))     # must be (1. nn output, 2. target)
        batch_loss.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    mean_loss=np.mean(np.array(batch_loss))
    print("Episode {} -> Train loss:{}".format(epoch,mean_loss))
    epp_train_loss.append(mean_loss)
    batch_loss=[]
    for step, (batch_x, batch_y) in enumerate(valid_loader): 
        b_x = Variable(batch_x)
        
        b_y = Variable(batch_y)

        prediction = net(b_x.to(device))     # input x and predict based on x
        loss = loss_func(prediction, b_y.to(device))     # must be (1. nn output, 2. target)
        batch_loss.append(loss.cpu().detach().numpy()) 
    mean_loss=np.mean(np.array(batch_loss))
    print("Episode {} -> Validation loss:{}".format(epoch,mean_loss))
    epp_valid_loss.append(mean_loss)


np.savetxt('Loss/TrainLoss_vel_D2_{}.txt'.format(index),epp_train_loss)
np.savetxt('Loss/ValidLoss_vel_D2_{}.txt'.format(index),epp_valid_loss)

