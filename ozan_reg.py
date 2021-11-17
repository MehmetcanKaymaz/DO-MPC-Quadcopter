import torch
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import StepLR
#System Check
#print(torch.cuda.is_available)
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#print (os.getcwd())
#--------------------------Initialization------------------------
#desired loss tanimlanacak
# Data Parameters
#batch_size = 16
batch_size_grid = [32]
# Model Parameters
input_size = 9
output_size = 4
# Train Parameters
learningRate = 0.1
epochs = 1000
#epochs_grid= [10,20,30]
#---------------------------DataLoader---------------------------
torch.manual_seed(1)    # reproducible
print("Loading datas ...")
matrix = np.loadtxt("Datas2/all_data.txt", dtype=np.float32)  #veriyi float32 cinsinden matrix degisenine aktariyor.
num_of_rows = len(matrix)
matrix = np.take(matrix,np.random.permutation(matrix.shape[0]),axis=0,out=matrix)
matrix = torch.from_numpy(matrix)

train_size = int(.8 * len(matrix))
train = matrix[0:train_size,:]
train_input =train[:,[0,1,2,3,4,5,6,7,8]]
train_label = train[:,[9,10,11,12]]

train_size  = len(train)
valid = matrix[train_size:num_of_rows,:]
valid_input = valid[:,[0,1,2,3,4,5,6,7,8]]
valid_label = valid[:,[9,10,11,12]]
valid_size  = len(valid)
#------------------------------Model-----------------------------
net = torch.nn.Sequential(
        torch.nn.Linear(input_size, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(32, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.ReLU(),
        #torch.nn.Dropout(0.2),
        torch.nn.Linear(32, output_size),
    )
net.to(device)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay= 0.001)


#for epochs in epochs_grid:
scheduler = StepLR(optimizer, step_size=int(epochs/4), gamma=0.1)
for batch_size in batch_size_grid:
    
    #--------------------Train Model--------------------------------
    # Make model ready to train
    net.train()
    train_dataset = Data.TensorDataset(train_input, train_label )
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True, num_workers=2,)
    valid_dataset = Data.TensorDataset(valid_input, valid_label)
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=int(batch_size/5),
        shuffle=True, num_workers=2,)
    epp_train_loss=[]
    epp_valid_loss=[]
    print("Training ....")

    for epoch in range(epochs+1):
        batch_loss=[]
        for step, (batch_x, batch_y) in enumerate(train_loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            net.train()
            prediction = net(b_x.to(device))     # input x and predict based on x
            #weighted loss calculation
            prediction[:,0:3] *= 10
            prediction[:,  3] /= 1
            b_y[:,0:3] *= 10
            b_y[:,  3] /= 1
            loss = loss_func(prediction, b_y.to(device))     # must be (1. nn output, 2. target)
            batch_loss.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        # Decay Learning Rate
        scheduler.step()
        mean_loss=np.mean(np.array(batch_loss))
        print("Epoch {}/{} ->      Train loss:{}  batchS: {}".format(epoch,epochs,mean_loss,batch_size))
        epp_train_loss.append(mean_loss)
        batch_loss=[]
        for step, (batch_x, batch_y) in enumerate(valid_loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            net.eval()
            prediction = net(b_x.to(device))     # input x and predict based on x
            #weighted loss calculation
            prediction[:,0:3] *= 10
            prediction[:,  3] /= 1
            b_y[:,0:3] *= 10
            b_y[:,  3] /= 1
            loss = loss_func(prediction, b_y.to(device))     # must be (1. nn output, 2. target)
            batch_loss.append(loss.cpu().detach().numpy())
        mean_loss=np.mean(np.array(batch_loss))
        print("Epoch {}/{} -> Validation loss:{}  batchS: {} ".format(epoch,epochs,mean_loss,batch_size ))
        epp_valid_loss.append(mean_loss)
        if epoch%200==0:
            torch.save(net.state_dict(), "Models/checkpoint_batchsize_ozan_final_{}_epoch_{}.pth".format(batch_size,epoch))
            print("checkpoint_for_epoch{}_batch_size{} saved!".format(epoch,batch_size))
    
    np.savetxt('TrainLoss_vel_ozan_final_batch_size_{}.txt'.format(epoch,batch_size),epp_train_loss)
    np.savetxt('ValidLoss_vel_ozan_final_batch_size_{}.txt'.format(epoch,batch_size),epp_valid_loss)