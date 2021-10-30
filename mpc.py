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


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 20,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 0.01,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    mterm = model.aux['cost'] # terminal cost
    lterm = model.aux['cost']

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(inp=np.array([[0.5], [0.5], [0.5]]))

    max_x = np.array([[np.pi/2], [np.pi/2], [np.pi/2]])
    max_u = np.array([[0.5], [0.5], [0.5]])

    # lower and upper bounds of the states
    mpc.bounds['lower','_x','angle'] = -max_x
    mpc.bounds['upper','_x','angle'] = max_x
     # lower and upper bounds of the states
    mpc.bounds['lower','_x','D_angle'] = -max_x
    mpc.bounds['upper','_x','D_angle'] = max_x   

    # lower bounds of the input
    mpc.bounds['lower','_u','inp'] = -max_u
    # upper bounds of the input
    mpc.bounds['upper','_u','inp'] =  max_u
    
    Iyy_values = np.array([1e-3])
    Ixx_values = np.array([1e-3])
    Izz_values = np.array([1e-3])
    #target = np.array([np.pi/4])

    mpc.set_uncertainty_values(Iyy=Iyy_values,Ixx=Ixx_values,Izz=Izz_values)

    mpc.setup()

    return mpc
