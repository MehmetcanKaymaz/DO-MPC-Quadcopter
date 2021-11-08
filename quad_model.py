import numpy as np


class Model:
    def __init__(self):
        self.x=np.zeros(12)
        self.u=np.zeros(4)
        self.x_dot=np.zeros(12)
        self.dt=.01
        self.m=.5
        self.g=9.81
        self.Ixx=1e-3
        self.Iyy=1e-3
        self.Izz=1e-3

    def reset(self,x):
        self.x=x
        self.x_dot=np.zeros(12)
        self.u=np.zers(4)

    def run_model(self,u):
        self.u=u
        self.__update_x_dot()
        self.__update_x()

        return self.x

    def __update_x(self):
        self.x+=self.x_dot*self.dt
    
    def __update_x_dot(self):
        self.x_dot[0]=self.x[3]*np.cos(self.x[8])-self.x[4]*np.sin(self.x[8])
        self.x_dot[1]=self.x[4]*np.cos(self.x[8])+self.x[3]*np.sin(self.x[8])
        self.x_dot[2]=self.x[5]
        self.x_dot[3]=-self.u[3]*np.sin(self.x[7])/self.m
        self.x_dot[4]=self.u[3]*np.sin(self.x[6])/self.m
        self.x_dot[5]=-self.g+self.u[3]*np.cos(self.x[6])*np.cos(self.x[7])/self.m
        self.x_dot[6]=self.x[9]+self.x[11]*(np.cos(self.x[6])*np.tan(self.x[7]))+self.x[10]*(np.sin(self.x[6])*np.tan(self.x[7]))
        self.x_dot[7]=self.x[10]*np.cos(self.x[6])-self.x[11]*np.sin(self.x[6])
        self.x_dot[8]= self.x[11]*np.cos(self.x[6])/np.cos(self.x[7])+self.x[10]*np.sin(self.x[6])/np.cos(self.x[7])
        self.x_dot[9]= (self.Iyy-self.Izz)*self.x[10]*self.x[11]/self.Ixx + self.u[0]/self.Ixx
        self.x_dot[10]=(self.Izz-self.Ixx)*self.x[9]*self.x[11]/self.Iyy + self.u[1]/self.Iyy
        self.x_dot[11]=(self.Ixx-self.Iyy)*self.x[10]*self.x[9]/self.Izz + self.u[2]/self.Izz




