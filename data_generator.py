import numpy as np 
import random
import os

epoch=5000
velx_lim=[-10,10]
vely_lim=[-10,10]
velz_lim=[-5,5]
errpsi_lim=[-1,1]

index=0

episode=1000

def calculate_random():
    x=random.uniform(velx_lim[0],velx_lim[1])
    y=random.uniform(vely_lim[0],vely_lim[1])
    z=random.uniform(velz_lim[0],velz_lim[1])
    psi=random.uniform(errpsi_lim[0],errpsi_lim[1])
    return [x,y,z,psi]

def arr_to_str(arr):
    s_cmd="{},{},{},{}".format(arr[0],arr[1],arr[2],arr[3])
    return s_cmd


for i in range(epoch):
    command="python mpc_vel.py --ref {} --idx {} --episode {}".format(arr_to_str(calculate_random()),index,episode)
    print(command)
    os.system(command)
    index+=1


