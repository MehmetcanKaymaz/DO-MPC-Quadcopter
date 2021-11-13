import numpy as np 
import random
import os
import multiprocessing as mp
import time


index=0
batch_size=[8,16,32,64]
learning_rate=[0.1,0.01,0.001,0.0001]



def run_code(batch_size,learning_rate,index):
    command="python train_mpc_vel.py --batch_size {} --learning_rate {} --index {}".format(batch_size,learning_rate,index)
    os.system(command)


if __name__=="__main__":
    ti=time.time()
    proceses=[]
    for j in range(4):
        for i in range(4):
            proc=mp.Process(target=run_code,args=(batch_size[j],learning_rate[i],index,))
            print("Proccess {} -> batch_size:{}  learning_rate {}".format(index,batch_size[j],learning_rate[i]))
            proceses.append(proc)
            proc.start()
            proc.join()
            index+=1
    tf=time.time()
    print("delta time:{}".format(tf-ti))

        



