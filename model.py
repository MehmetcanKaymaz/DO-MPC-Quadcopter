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

    # States struct (optimization variables): roll, pitch, yaw
    angle    = model.set_variable(var_type='_x', var_name='angle', shape=(3,1)) # roll, pitch and yaw 
    D_angle  = model.set_variable(var_type='_x', var_name='D_angle', shape=(3,1)) # first derivation of roll, pitch and yaw angle
    #DD_angle = model.set_variable(var_type='_x', var_name='DD_angle', shape=(3,1)) # second derivation of roll, pitch and yaw angle
    
    # Input struct (optimization variables):
    inp = model.set_variable(var_type='_u', var_name='inp', shape=(3,1)) # u1, u2, u3
   
    # uncertain parameters:
    Ixx = model.set_variable('_p',  'Ixx')
    Iyy = model.set_variable('_p',  'Iyy')
    Izz = model.set_variable('_p',  'Izz')

    DD_angle_next = vertcat( 
                            inp[0]/Ixx ,  
                            inp[1]/Iyy , 
                            inp[2]/Izz 
    ) 

    # Differential equations
    model.set_rhs('angle', D_angle)
    model.set_rhs('D_angle', DD_angle_next)
    
    # Cost
    cost=pow(np.pi/4-angle[0],2)+pow(np.pi/8-angle[1],2)+pow(-np.pi/8-angle[2],2)

    model.set_expression('cost', cost)

    # Build the model
    model.setup()

    return model
