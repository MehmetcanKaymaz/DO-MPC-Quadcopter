import numpy as np 
import random
import os

epoch=2000
errx_lim=[0,15]
erry_lim=[-10,10]
errz_lim=[-5,5]
errpsi_lim=[-np.pi,np.pi]

v_lim=[1,30]

index=0

episode=1000

def calculate_random():
    x=random.uniform(errx_lim[0],errx_lim[1])
    y=random.uniform(erry_lim[0],erry_lim[1])
    z=random.uniform(errz_lim[0],errz_lim[1])
    psi=random.uniform(errpsi_lim[0],errpsi_lim[1])
    v=random.uniform(v_lim[0],v_lim[1])
    return [x,y,z,psi,v]

def arr_to_str(arr):
    s_cmd="{},{},{},{},{}".format(arr[0],arr[1],arr[2],arr[3],arr[4])
    return s_cmd


for i in range(epoch):
    command="python main.py --ref {} --idx {} --episode {}".format(arr_to_str(calculate_random()),index,episode)
    os.system(command)
    index+=1
    print("*\n"*100)


