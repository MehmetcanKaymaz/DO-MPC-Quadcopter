import numpy as np

class Traj_Planner:
    def __init__(self):
        self.T=5
        self.cx=np.zeros(4)
        self.cy=np.zeros(4)
        self.cz=np.zeros(4)
        self.cpsi=np.zeros(4)

        self.x_initiral=np.zeros(4)
        self.x_final=np.zeros(4)
        self.v_initial=np.zeros(4)
        self.v_final=np.zeros(4)

        self.xm=np.zeros(4)
        self.ym=np.zeros(4)
        self.zm=np.zeros(4)
        self.psim=np.zeros(4)

    def __calculate_A(self):
        T=self.T
        A=np.matrix([[1,0,0,0],[1,T,pow(T,2),pow(T,3)],[0,1,0,0],[0,1,2*T,3*pow(T,2)]])
        return A

    def __calculate_c(self,x):
        A=self.__calculate_A()
        inverse_A=np.linalg.inv(A)
        x_matrix=np.matrix([[x[0]],[x[1]],[x[2]],[x[3]]])
        #print("x_matrix:{}".format(x_matrix))
        c=np.array(np.matmul(inverse_A,x_matrix))
        c_a=np.zeros(4)
        for i in range(4):
            c_a[i]=c[i][0]
        return c_a
    def __calculate_xm(self):
        self.xm=np.array([self.x_initiral[0],self.x_final[0],self.v_initial[0],self.v_final[0]])
        self.ym=np.array([self.x_initiral[1],self.x_final[1],self.v_initial[1],self.v_final[1]])
        self.zm=np.array([self.x_initiral[2],self.x_final[2],self.v_initial[2],self.v_final[2]])
        self.psim=np.array([self.x_initiral[3],self.x_final[3],self.v_initial[3],self.v_final[3]])
        

    def find_traj(self,x_initial,x_final,v_initial,v_final,T):
        self.x_initiral=x_initial
        self.x_final=x_final
        self.v_initial=v_initial
        self.v_final=v_final
        self.T=T
        self.__calculate_xm()
        self.cx=self.__calculate_c(self.xm)
        self.cy=self.__calculate_c(self.ym)
        self.cz=self.__calculate_c(self.zm)
        self.cpsi=self.__calculate_c(self.psim)
        #print("cx : {}".format(self.cx))
        

    def __calculate_ref(self,c,t):
        x=c[0]+c[1]*t+c[2]*pow(t,2)+c[3]*pow(t,3)
        return x
    def __calculate_vel(self,c,t):
        x=c[1]+c[2]*t+c[3]*2*pow(t,2)
        return x

    def get_target(self,t):
        xt=self.__calculate_ref(self.cx,t)
        yt=self.__calculate_ref(self.cy,t)
        zt=self.__calculate_ref(self.cz,t)
        psit=self.__calculate_ref(self.cpsi,t)
        
        return np.array([xt,yt,zt,psit])

    def get_vel(self,t):
        vxd=self.__calculate_vel(self.cx,t)
        vyd=self.__calculate_vel(self.cy,t)
        vzd=self.__calculate_vel(self.cz,t)
        rd=self.__calculate_vel(self.cpsi,t)

        return np.array([vxd,vyd,vzd,rd])


    
