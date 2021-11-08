import numpy as np
import torch


class Controller:
    def __init__(self):
        self.u=np.zeros(4)
        self.x=np.zeros(12)
        self.x_t=np.zeros(5)
        self.x_nn=np.zeros(13)
        self.net=torch.nn.Sequential(
                    torch.nn.Linear(13, 256),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(256, 4),
                )
        self.net.load_state_dict(torch.load("Models/checkpoint6500.pth"))
        self.net.eval()  

    def __conf_inputs(self):
        self.x_nn[0]=self.x_t[4]
        for i in range(3):
            self.x_nn[i+1]=self.x_t[i]-self.x[i]
        for i in range(5):
            self.x_nn[i+4]=self.x[i+3]
        self.x_nn[9]=self.x_t[3]-self.x[8]
        for i in range(3):
            self.x_nn[i+10]=self.x[i+9]
    
    def __run_nn(self):
        self.u=self.net(torch.from_numpy(np.array(self.x_nn,np.float32))).cpu().detach().numpy()
        return self.u


    def run_controller(self,x,x_t):
        self.x=x
        self.x_t=x_t
        self.__conf_inputs()
        self.u=self.__run_nn()
        
        return self.u


