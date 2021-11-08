import numpy as np
from controller import Controller
from quad_model import Model
import matplotlib.pyplot as plt



quad=Model()
controller=Controller()

x0=quad.x

T=5.0
dt=1e-2
N=int(T/dt)
t=np.linspace(0,T,N)

xd=5*np.ones(N)
yd=3*np.ones(N)
zd=2*np.ones(N)
psid=(np.pi/4)*np.ones(N)
V_max=5

state_arr=np.zeros((N,12))
u_list=np.zeros((N,4))

#u_arr=np.loadtxt("Datas/data0.txt")[:,[13,14,15,16]]


for i in range(N):
    u=controller.run_controller(x=x0,x_t=[xd[i],yd[i],zd[i],psid[i],V_max])
    #u_list[i,:]=u_arr[i]
    u_list[i,:]=u
    next_x=quad.run_model(u_list[i,:])
    state_arr[i,:]=x0
    x0=next_x

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

plt.plot(t,state_arr[:,8])
plt.plot(t,psid)
plt.xlabel("time(s)")
plt.ylabel("psi(rad)")
plt.legend(["psi_real psi_desered"])
plt.show()



