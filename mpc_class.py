import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
import time
from mpc import template_mpc
from mpc_sim import template_simulator



class MPC_controller:
    def __init__(self):
        self.info=""
        self.ref_vel=np.zeros(4)
        self.mpc=None
        self.x0=np.zeros(9)

    
    def __template_model(self,symvar_type='SX'):
        """
        --------------------------------------------------------------------------
        template_model: Variables / RHS / AUX
        --------------------------------------------------------------------------
        """
        model_type = 'continuous' # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type, symvar_type)

        # States struct (optimization variables): roll, pitch, yaw
        Velocity= model.set_variable(var_type='_x', var_name='Velocity', shape=(3,1))
        Attitude= model.set_variable(var_type='_x', var_name='Attitude', shape=(3,1)) # roll, pitch and yaw 
        Rate= model.set_variable(var_type='_x', var_name='Rate', shape=(3,1)) # first derivation of roll, pitch and yaw angle

        # Input struct (optimization variables):
        inp = model.set_variable(var_type='_u', var_name='inp', shape=(4,1)) # u1, u2, u3
    
        # uncertain parameters:
        Ixx = model.set_variable('_p',  'Ixx')
        Iyy = model.set_variable('_p',  'Iyy')
        Izz = model.set_variable('_p',  'Izz')
        g=model.set_variable('_p',  'g')
        m=model.set_variable('_p',  'm')

        Velocity_dot=vertcat( 
                                -inp[3]*(np.sin(Attitude[1]))/m,  
                                inp[3]*(np.sin(Attitude[0]))/m, 
                                -g+inp[3]*np.cos(Attitude[0])*np.cos(Attitude[1])/m
        )

        Rate_dot = vertcat( 
                                (Iyy-Izz)*Rate[1]*Rate[2]/Ixx + inp[0]/Ixx ,  
                                (Izz-Ixx)*Rate[0]*Rate[2]/Iyy + inp[1]/Iyy , 
                                (Ixx-Iyy)*Rate[1]*Rate[0]/Izz + inp[2]/Izz 
        ) 
        
        Attitude_dot=vertcat(Rate[0]+Rate[2]*(np.cos(Attitude[0])*np.tan(Attitude[1]))+Rate[1]*(np.sin(Attitude[0])*np.tan(Attitude[1])),
        Rate[1]*np.cos(Attitude[0])-Rate[2]*np.sin(Attitude[0]),
        Rate[2]*np.cos(Attitude[0])/np.cos(Attitude[1])+Rate[1]*np.sin(Attitude[0])/np.cos(Attitude[1]))

        # Differential equations
        model.set_rhs('Attitude', Attitude_dot)
        model.set_rhs('Rate', Rate_dot)
        model.set_rhs('Velocity', Velocity_dot)

        cost=.5*pow(1*self.ref_vel[0]-Velocity[0],2)+.5*pow(1*self.ref_vel[1]-Velocity[1],2)+.7*pow(1*self.ref_vel[2]-Velocity[2],2)+5*pow(self.ref_vel[3]-Attitude[2],2)

        model.set_expression('cost', cost)

        # Build the model
        model.setup()

        return model
    
    def __set_model(self,x,x_t):
        self.ref_vel=np.array(x_t)
        model = self.__template_model()
        self.mpc = template_mpc(model)
        simulator = template_simulator(model)
        estimator = do_mpc.estimator.StateFeedback(model)

        self.x0 = np.array(x).reshape(-1,1) # roll, pitch, yaw and its derivatives
        self.mpc.x0 = self.x0
        simulator.x0 = self.x0
        estimator.x0 = self.x0

        self.mpc.set_initial_guess()
        

    def __configure_u(self,u):
        for i in range(3):
            u[i]=u[i]*10+5

        return u

    def run_controller(self,x,x_t):
        self.__set_model(x=x,x_t=x_t)
        u0 = self.mpc.make_step(self.x0)
        u0=self.__configure_u(u=u0)
        return np.array(u0).reshape((1,4))[0]


    
    
    
    