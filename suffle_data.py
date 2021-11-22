import numpy as np
import random


data=np.loadtxt('Dagger-All-Data/Dagger_D0_D1_D2.txt')

N=len(data[:,0])

suffle_data=np.zeros((N,13))



for i in range(N):
    index=random.randint(0,N-i-1)
    suffle_data[i,:]=data[index,:]
    data=np.delete(data,(index),axis=0)

np.savetxt('Dagger-All-Data/Dagger_D0_D1_D2_s.txt',suffle_data)