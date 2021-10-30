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

from model import template_model
from mpc import template_mpc
from sim import template_simulator

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
x0 = np.pi*np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1,1) # roll, pitch, yaw and its derivatives

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

for k in range(300):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    

    if show_animation:
        mpc_plot.plot_results()
        mpc_plot.plot_predictions()
        mpc_plot.reset_axes()
        plt.show()
        plt.pause(0.01)



input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')
