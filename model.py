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
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)


    # States struct (optimization variables):
    Theta = model.set_variable('_x',  'Theta')  # bio mass
    q_rate = model.set_variable('_x',  'q_rate')  # Substrate
    Roll = model.set_variable('_x',  'Roll')
    p_rate = model.set_variable('_x',  'p_rate')
    Yaw = model.set_variable('_x',  'Yaw')
    r_rate = model.set_variable('_x',  'r_rate')

    #theta_target=model.set_variable('_x','Theta-target')
    # Input struct (optimization variables):
    inp1 = model.set_variable('_u',  'inp1')
    inp2 = model.set_variable('_u',  'inp2')
    inp3 = model.set_variable('_u',  'inp3')

    # Fixed parameters:
    Iyy = model.set_variable('_p',  'Iyy')
    Ixx = model.set_variable('_p',  'Ixx')
    Izz = model.set_variable('_p',  'Izz')
    #theta_target=model.set_variable('_p','Thetatarget')
    
    cost=pow(np.pi/4-Theta,2)+pow(np.pi/8-Roll,2)+pow(-np.pi/8-Yaw,2)

    model.set_expression('cost', cost)
    #model.set_expression('Theta-current', Theta)

    # Differential equations
    model.set_rhs('Theta', q_rate)
    model.set_rhs('q_rate', inp1/Iyy)
    model.set_rhs('Roll', p_rate)
    model.set_rhs('p_rate', inp2/Ixx)
    model.set_rhs('Yaw', r_rate)
    model.set_rhs('r_rate', inp3/Izz)   

    # Build the model
    model.setup()

    return model
