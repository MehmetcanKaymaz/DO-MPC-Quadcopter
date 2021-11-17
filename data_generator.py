import numpy as np 
import random
import os
import multiprocessing as mp
import time

epoch=25
velx_lim=[-15,15]
vely_lim=[-3,3]
velz_lim=[-2,2]
errpsi_lim=[-np.pi,np.pi]

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

def run_code(index):
    command="python dagger_data_gen.py --ref={} --idx {} --episode {}".format(arr_to_str(calculate_random()),index,episode)
    os.system(command)


if __name__=="__main__":
    ti=time.time()
    for i in range(epoch):
        proceses=[]
        for j in range(4):
            proc=mp.Process(target=run_code,args=(index,))
            proceses.append(proc)
            proc.start()
            index+=1
        for proc in proceses:
            proc.join()
    tf=time.time()
    print("delta time:{}".format(tf-ti))

        



