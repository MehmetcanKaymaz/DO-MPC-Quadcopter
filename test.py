import numpy as np
from mpc_class import MPC_controller
import time


controller=MPC_controller()


x0=np.zeros(9)
xt=np.array([2,2,2,np.pi/4])

ti=time.time()
print(controller.run_controller(x=x0,x_t=xt))
tf=time.time()
print("Delta T:{}".format(tf-ti))