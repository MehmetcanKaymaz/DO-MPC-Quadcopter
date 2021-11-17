import numpy as np
from controller import Controller
from quad_model import Model
import matplotlib.pyplot as plt



quad=Model()
controller=Controller()

x0=quad.x

T=1.0
dt=1e-2
N=int(T/dt)
t=np.linspace(0,T,N)

"""
xd=5*np.ones(N)
yd=1*np.ones(N)
zd=.5*np.ones(N)
psid=(np.pi/4)*np.ones(N)/np.pi

state_arr=np.zeros((N,9))


"""
data_arr=np.loadtxt("Datas/data780.txt")
inputs=data_arr[:,0:9]
outputs=data_arr[:,9:13]
N=len(inputs[:,0])
u_list=np.zeros((N,4))
t=np.linspace(0,int(N/100),N)
mse_arr=np.zeros(N)
for i in range(N):
    u=controller.run_controller_nn(inputs[i,:])
    #u_list[i,:]=u_arr[i]
    u_list[i,:]=u
    u_err=0
    for j in range(4):
        u_err+=np.sqrt(pow(outputs[i,j]-u[j],2))
    mse_arr[i]=u_err
    """next_x=quad.run_model(u_list[i,:])
    state_arr[i,:]=x0[3:12]
    x0=next_x"""
"""
plt.plot(t,state_arr[:,0])
plt.plot(t,xd)
plt.xlabel("time(s)")
plt.ylabel("x(m)")
plt.legend(["x_real x_desered"])
plt.show()

plt.plot(t,state_arr[:,1])
plt.plot(t,yd)
plt.xlabel("time(s)")
plt.ylabel("y(m)")
plt.legend(["y_real y_desered"])
plt.show()

plt.plot(t,state_arr[:,2])
plt.plot(t,zd)
plt.xlabel("time(s)")
plt.ylabel("z(m)")
plt.legend(["z_real z_desered"])
plt.show()

plt.plot(t,state_arr[:,5])
plt.plot(t,psid)
plt.xlabel("time(s)")
plt.ylabel("psi(rad)")
plt.legend(["psi_real psi_desered"])
plt.show()

"""

fig, axs = plt.subplots(3, 2)
axs[0,0].plot(t,u_list[:,0])
axs[0,0].plot(t,outputs[:,0])
axs[0,0].legend(["u_real"," u_desered"])
axs[0,0].set_title("U1")

axs[0,1].plot(t,u_list[:,1])
axs[0,1].plot(t,outputs[:,1])
axs[0,1].legend(["u_real"," u_desered"])
axs[0,1].set_title("U2")

axs[1,0].plot(t,u_list[:,2])
axs[1,0].plot(t,outputs[:,2])
axs[1,0].legend(["u_real"," u_desered"])
axs[1,0].set_title("U3")

axs[1,1].plot(t,u_list[:,3])
axs[1,1].plot(t,outputs[:,3])
axs[1,1].legend(["u_real"," u_desered"])
axs[1,1].set_title("U4")

axs[2,1].plot(t,mse_arr)
axs[2,1].legend(["mse"])
axs[2,1].set_title("mean square error")
plt.show()



