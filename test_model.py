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


xd=5*np.ones(N)
yd=1*np.ones(N)
zd=.5*np.ones(N)
psid=(np.pi/4)*np.ones(N)/np.pi

state_arr=np.zeros((N,9))


""""""
data_arr=np.loadtxt("Datas/data750.txt")
inputs=data_arr[:,0:9]
outputs=data_arr[:,9:13]
N=len(inputs[:,0])
u_list=np.zeros((N,4))
t=np.linspace(0,int(N/100),N)

for i in range(N):
    u=controller.run_controller_nn(inputs[i,:])
    #u_list[i,:]=u_arr[i]
    u_list[i,:]=u
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
plt.plot(t,u_list[:,0])
plt.plot(t,outputs[:,0])
plt.xlabel("time(s)")
plt.ylabel("u1(m)")
plt.legend(["u_real"," u_desered"])
plt.show()

plt.plot(t,u_list[:,1])
plt.plot(t,outputs[:,1])
plt.xlabel("time(s)")
plt.ylabel("u2(m)")
plt.legend(["u_real"," u_desered"])
plt.show()

plt.plot(t,u_list[:,2])
plt.plot(t,outputs[:,2])
plt.xlabel("time(s)")
plt.ylabel("u3(m)")
plt.legend(["u_real"," u_desered"])
plt.show()

plt.plot(t,u_list[:,3])
plt.plot(t,outputs[:,3])
plt.xlabel("time(s)")
plt.ylabel("u4(rad)")
plt.legend(["u_real"," u_desered"])
plt.show()



