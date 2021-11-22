import numpy as np
from controller import Controller
from quad_model import Model
import matplotlib.pyplot as plt
from mpc_class import MPC_controller
import argparse

parser = argparse.ArgumentParser(description='MPC')
parser.add_argument('--ref', default="10.0,2.0,-1.0,0.3" , type=str,
                    help='target vel')
parser.add_argument('--idx', default=0 , type=int,
                    help='index')
parser.add_argument('--episode', default=500 , type=int,
                    help='episode')
args = parser.parse_args()

file_idx=args.idx
episode=args.episode
ref_vel_arr=args.ref.split(",")
ref_vel=[]



for pose in ref_vel_arr:
    ref_vel.append(float(pose))



quad=Model()
quad2=Model()
controller=Controller()
mpc_controller=MPC_controller()

def to_dataset(x,u):
    x_data=np.zeros(13)
    for i in range(9):
        x_data[i]=x[i]
    for i in range(4):
        x_data[i+9]=u[i]
    return x_data

def conf_u(u):
    for i in range(3):
        u[i]=(u[i]-5)/10

    return u

x0=quad.x
x02=quad2.x

T=2
dt=1e-2
N=int(T/dt)
t=np.linspace(0,T,N)

xd=ref_vel[0]*np.linspace(0,1,N)
yd=ref_vel[1]*np.linspace(0,1,N)
zd=ref_vel[2]*np.linspace(0,1,N)
psid=ref_vel[3]*np.linspace(0,1,N)

state_arr=np.zeros((N,9))
state_arr_mpc=np.zeros((N,9))
u_list_nn=np.zeros((N,4))
u_list_mpc=np.zeros((N,4))


mse_arr=np.zeros(N)
index=0
Data_arr=[]
for i in range(N):
    xt=[xd[i],yd[i],zd[i],psid[i]]
    x0=x0[3:12]
    x02=x02[3:12]
    u_nn=controller.run_controller(x=x0,x_t=xt)
    #u_mpc=mpc_controller.run_controller(x=x02,x_t=xt)
    u_list_mpc[i,:]=[0,0,0,0]
    u_list_nn[i,:]=u_nn
    #print("u_mpc:{}  u_nn:{}".format(u_mpc,u_nn))
    err=0
    for j in range(3):
        err+=0#np.sqrt(pow(u_nn[j]-u_mpc[j],2))
    if err>.3:
        #Data_arr.append(to_dataset(x=controller.x_nn,u=u_mpc))       
        mse_arr[i]=err
        x0=quad.run_model(conf_u(u_nn))
        #x02=quad2.run_model(conf_u(u_mpc))
        index+=1
    else:
        x0=quad.run_model(conf_u(u_nn))
        #x02=quad2.run_model(conf_u(u_mpc))
        mse_arr[i]=0
    state_arr[i,:]=x0[3:12]
    #state_arr_mpc[i,:]=x02[3:12]
    

#np.savetxt("Datas_Dagger_1/data_{}.txt".format(file_idx),np.array(Data_arr))
#print("index={}".format(index))

fig, axs = plt.subplots(3, 2)
axs[0,0].plot(t,u_list_nn[:,0])
axs[0,0].plot(t,u_list_mpc[:,0])
axs[0,0].legend(["u_nn"," u_mpc"])
axs[0,0].set_title("U1")

axs[0,1].plot(t,u_list_nn[:,1])
axs[0,1].plot(t,u_list_mpc[:,1])
axs[0,1].legend(["u_nn"," u_mpc"])
axs[0,1].set_title("U2")

axs[1,0].plot(t,u_list_nn[:,2])
axs[1,0].plot(t,u_list_mpc[:,2])
axs[1,0].legend(["u_nn"," u_mpc"])
axs[1,0].set_title("U3")

axs[1,1].plot(t,u_list_nn[:,3])
axs[1,1].plot(t,u_list_mpc[:,3])
axs[1,1].legend(["u_nn"," u_mpc"])
axs[1,1].set_title("U4")

axs[2,1].plot(t,mse_arr)
axs[2,1].legend(["mse"])
axs[2,1].set_title("mean square error")
plt.show()

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(t,xd)
axs[0,0].plot(t,state_arr[:,0])
axs[0,0].plot(t,state_arr_mpc[:,0])
axs[0,0].legend(["x_t"," x_nn","x_mpc"])

axs[0,1].plot(t,yd)
axs[0,1].plot(t,state_arr[:,1])
axs[0,1].plot(t,state_arr_mpc[:,1])
axs[0,1].legend(["y_t"," y_nn","y_mpc"])

axs[1,0].plot(t,zd)
axs[1,0].plot(t,state_arr[:,2])
axs[1,0].plot(t,state_arr_mpc[:,2])
axs[1,0].legend(["z_t"," y_nn","xy_mpc"])

axs[1,1].plot(t,psid)
axs[1,1].plot(t,state_arr[:,5])
axs[1,1].plot(t,state_arr_mpc[:,5])
axs[1,1].legend(["psi_t"," psi_nn","psi_mpc"])


plt.show()



