import numpy as np
import torch


class Controller:
    def __init__(self):
        self.u=np.zeros(4)
        self.x=np.zeros(9)
        self.x_t=np.zeros(4)
        self.x_nn=np.zeros(9)
        self.net=torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
    )
        self.net.load_state_dict(torch.load("Models/checkpoint_vel_1000.pth"))
        self.net.eval()  

    def __conf_inputs(self):
        self.x_nn[0]=self.x_t[0]-self.x[0]
        self.x_nn[1]=self.x_t[1]-self.x[1]
        self.x_nn[2]=self.x_t[2]-self.x[2]
        self.x_nn[3]=self.x[3]
        self.x_nn[4]=self.x[4]
        self.x_nn[5]=self.x_t[3]-self.x[5]
        self.x_nn[6]=self.x[6]
        self.x_nn[7]=self.x[7]
        self.x_nn[8]=self.x[8]

    
    def __run_nn(self):
        self.u=self.net(torch.from_numpy(np.array(self.x_nn,np.float32))).cpu().detach().numpy()
        return self.u

    def __calculate_u(self,u):
        for i in range(3):
            u[i]=(u[i]-5)/10
        return u


    def run_controller(self,x,x_t):
        self.x=x
        self.x_t=x_t
        self.__conf_inputs()
        self.u=self.__run_nn()
        self.u=self.__calculate_u(self.u)
        
        return self.u

    def run_controller_nn(self,x):
        self.x_nn=x
        print(self.x_nn)
        self.u=self.__run_nn()

        return self.u


