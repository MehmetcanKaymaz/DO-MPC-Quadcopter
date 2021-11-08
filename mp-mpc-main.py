#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import argparse
#from model import template_model
from mpc import template_mpc
from mpc_sim import template_simulator
from traj_planner import Traj_Planner

"""parser = argparse.ArgumentParser(description='MPC')
parser.add_argument('--ref', default="4.0,4.0,4.0,0.5" , type=str,
                    help='target pose')
parser.add_argument('--idx', default=0 , type=int,
                    help='index')
parser.add_argument('--episode', default=500 , type=int,
                    help='episode')
args = parser.parse_args()

file_idx=args.idx
episode=args.episode
ref_pose_arr=args.ref.split(",")
ref_pose=[]"""



"""for pose in ref_pose_arr:
    ref_pose.append(float(pose))
const_vel=ref_pose[4]"""

t=0.0

"""traj=Traj_Planner()
x_initial=[0,0,0,0]
x_final=[5,0,0,0]
v_initial=[0,0,0,0]
v_final=[0,0,0,0]
T=2
traj.find_traj(x_initial,x_final,v_initial,v_final,T)"""

T=5
dt=.01
N=int(T/dt)
t=np.linspace(0,T,N)
xd=5*np.sin(2*np.pi*t/5)
yd=5*np.cos(2*np.pi*t/5)
zd=2*t
psid=0
index=0
target_pose_arr=np.zeros((N,3))

def calculate_cost(inputs):
    pose_err=[inputs[1],inputs[2],inputs[3]]
    dis_err=norm_3d(pose_err)
    cost=dis_err#+10*inputs[9]
    return cost

def states_to_nn(xd,x,u):
    inputs=[]
    inputs.append(xd[4])
    for i in range(3):
        inputs.append(xd[i]-x[i][0])
    for i in range(5):
        inputs.append(x[i+3][0])
    inputs.append(xd[3]-x[8][0])
    for i in range(3):
        inputs.append(x[i+9][0])
    for i in range(4):
        inputs.append(u[i][0])
    return inputs    

def norm_3d(vec):
    return np.sqrt(pow(vec[0],2)+pow(vec[1],2)+pow(vec[2],2))

def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables): roll, pitch, yaw
    Pose= model.set_variable(var_type='_x', var_name='Pose', shape=(3,1))
    Velocity= model.set_variable(var_type='_x', var_name='Velocity', shape=(3,1))
    Attitude    = model.set_variable(var_type='_x', var_name='Attitude', shape=(3,1)) # roll, pitch and yaw 
    Rate  = model.set_variable(var_type='_x', var_name='Rate', shape=(3,1)) # first derivation of roll, pitch and yaw angle
    #DD_angle = model.set_variable(var_type='_x', var_name='DD_angle', shape=(3,1)) # second derivation of roll, pitch and yaw angle
    
    # Input struct (optimization variables):
    inp = model.set_variable(var_type='_u', var_name='inp', shape=(4,1)) # u1, u2, u3
   
    # uncertain parameters:
    Ixx = model.set_variable('_p',  'Ixx')
    Iyy = model.set_variable('_p',  'Iyy')
    Izz = model.set_variable('_p',  'Izz')
    g=model.set_variable('_p',  'g')
    m=model.set_variable('_p',  'm')

    """
    Velocity_dot=vertcat( 
                            Velocity[1]*Rate[2]-Velocity[2]*Rate[1]+g*np.sin(Attitude[1]),  
                            Velocity[2]*Rate[0]-Velocity[0]*Rate[2]+g*np.cos(Attitude[1])*np.sin(Attitude[0]), 
                            Velocity[0]*Rate[1]-Velocity[1]*Rate[0]-g*np.cos(Attitude[1])*np.sin(Attitude[0])+inp[3]/m
    )"""
    Pose_dot=vertcat( 
                            Velocity[0]*np.cos(Attitude[2])-Velocity[1]*np.sin(Attitude[2]),  
                            Velocity[1]*np.cos(Attitude[2])+Velocity[0]*np.sin(Attitude[2]), 
                            Velocity[2])
    Velocity_dot=vertcat( 
                            -inp[3]*(np.sin(Attitude[1]))/m,  
                            inp[3]*(np.sin(Attitude[0]))/m, 
                            -g+inp[3]*np.cos(Attitude[0])*np.cos(Attitude[1])/m
    )
    """
    Velocity_dot=vertcat( 
                            -inp[3]*(np.sin(Attitude[0])*np.sin(Attitude[2])+np.cos(Attitude[0])*np.cos(Attitude[2])*np.sin(Attitude[1]))/m,  
                            -inp[3]*(np.cos(Attitude[0])*np.sin(Attitude[2])*np.sin(Attitude[1])-np.cos(Attitude[2])*np.sin(Attitude[1]))/m, 
                            g-inp[3]*np.cos(Attitude[0])*np.cos(Attitude[1])/m
    )"""
    """
    Rate_dot = vertcat( 
                            inp[0]/Ixx ,  
                            inp[1]/Iyy , 
                            inp[2]/Izz 
    )
    """ 
    Rate_dot = vertcat( 
                            (Iyy-Izz)*Rate[1]*Rate[2]/Ixx + inp[0]/Ixx ,  
                            (Izz-Ixx)*Rate[0]*Rate[2]/Iyy + inp[1]/Iyy , 
                            (Ixx-Iyy)*Rate[1]*Rate[0]/Izz + inp[2]/Izz 
    ) 
    
    Attitude_dot=vertcat(Rate[0]+Rate[2]*(np.cos(Attitude[0])*np.tan(Attitude[1]))+Rate[1]*(np.sin(Attitude[0])*np.tan(Attitude[1])),
    Rate[1]*np.cos(Attitude[0])-Rate[2]*np.sin(Attitude[0]),
    Rate[2]*np.cos(Attitude[0])/np.cos(Attitude[1])+Rate[1]*np.sin(Attitude[0])/np.cos(Attitude[1]))
    """
    Attitude_dot=vertcat(Rate[0],
    Rate[1],
    Rate[2]
    )"""
    # Differential equations
    model.set_rhs('Pose', Pose_dot)
    model.set_rhs('Attitude', Attitude_dot)
    model.set_rhs('Rate', Rate_dot)
    model.set_rhs('Velocity', Velocity_dot)

    #ref_pose=[5,5,5]
    """
    current_pose=[Pose[0],Pose[1],Pose[2]]

    Err_pose=[ref_pose[0]-current_pose[0],ref_pose[1]-current_pose[1],ref_pose[2]-current_pose[2]]

    Distance_err=np.sqrt(pow(Err_pose[0],2)+pow(Err_pose[1],2)+pow(Err_pose[2],2))

    statu=if_else(Distance_err<1,True,False)

    ref_pose=if_else(statu==True,[0,0,0],[5,5,5])"""

    target_pose=[xd[index],yd[index],zd[index]]
    target_pose_arr[index,:]=target_pose
    target_vel_body=[target_pose[0]-Pose[0],target_pose[1]-Pose[1],target_pose[2]-Pose[2]]

    

    target_vel=[target_vel_body[0]*np.cos(Attitude[2])+target_vel_body[1]*np.sin(Attitude[2]),target_vel_body[1]*np.cos(Attitude[2])-target_vel_body[0]*np.sin(Attitude[2]),target_vel_body[2]]
    
    for i in range(3):
        target_vel[i]*=3

    var_vel=norm_3d(target_vel)

    V_ratio=if_else(var_vel>30,var_vel/30,1)

    for i in range(3):
        target_vel[i]=target_vel[i]/V_ratio
    
    
    # Cost
    cost=.1*pow(1*target_vel[0]-Velocity[0],2)+.1*pow(1*target_vel[1]-Velocity[1],2)+.1*pow(1*target_vel[2]-Velocity[2],2)+5*pow(psid*np.pi-Attitude[2],2)

    model.set_expression('cost', cost)

    # Build the model
    model.setup()

    return model

""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""
x0 = np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1,1) # roll, pitch, yaw and its derivatives
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

"""
Setup graphic:
"""

#color = plt.rcParams['axes.prop_cycle'].by_key()['color']
rcParams['axes.grid'] = True
fig, ax = plt.subplots(5,1, sharex=True, figsize=(10, 9))
mpc_plot = do_mpc.graphics.Graphics(mpc.data)

ax[0].set_title('Pose:')
ax[0].set_ylabel('Pose\n[m]')
mpc_plot.add_line('_x', 'Pose', ax[0]) 
ax[0].legend(['x', 'y', 'z'])

ax[1].set_title('Velocity:')
ax[1].set_ylabel('Velocity\n[m/s]')
mpc_plot.add_line('_x', 'Velocity', ax[1]) 
ax[1].legend(['Vel_x', 'Vel_y', 'Vel_z'])

ax[2].set_title('Rotation Angles:')
ax[2].set_ylabel('angle\n[rad]')
mpc_plot.add_line('_x', 'Attitude', ax[2]) 
ax[2].legend(['Roll', 'Pitch', 'Yaw'])

ax[3].set_title('Angular Velocity:')
ax[3].set_ylabel('angle velocity\n[rad/s2]')
mpc_plot.add_line('_x', 'Rate', ax[3])
ax[3].legend(['Roll Rate', 'Pitch Rate', 'Yaw Rate'])

ax[4].set_title('Inputs')
ax[4].set_ylabel('inputs')
mpc_plot.add_line('_u', 'inp', ax[4])
ax[4].legend(['input 1', 'input 2', 'input 3','input 4'])


fig.tight_layout()
plt.ion()



"""
Run MPC main loop:
"""
for k in range(N):
    index+=1
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        mpc_plot.plot_results()
        mpc_plot.plot_predictions()
        mpc_plot.reset_axes()
        plt.show()
        plt.pause(0.01)

input("Press Enter")
for i in range(N):
    print(target_pose_arr[i,:])
# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')
