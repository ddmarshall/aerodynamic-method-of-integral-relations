import numpy as np
import scipy as sci
# from scipy import optimize
from scipy import integrate


class Frozen_Cone_Flow:


    def __init__(self, Mach, gamma, Beta):

        self.Mach = Mach
        self.gamma = gamma
        self.Beta = np.deg2rad(Beta)


    def one_strip_solve(self, full_output: bool):
        """
        Frozen conical flow using 1-stirp Approximation Integral Relations 
        Refering Jerry South 1963

        -Created 11/5/2022 Eddie Hsieh

        Args:
            M_inf (float): Freestream Mach Number
            Beta (float): Shock Wave angle [deg]
            Gamma (float): Specific Heat Ratio (Constant)

        Returns:
            theta (float): Half Cone Angle [deg]
        """
        
        # Call Class Variables
        M_inf = self.Mach
        gamma = self.gamma
        Beta = self.Beta

        m = (M_inf**2)*(np.sin(Beta))**2
        a = 5*(m-1)/(6*M_inf**2)

        # Density at strip_delta
        rho_del = (6*m)/(5 + m)

        # Pressure at strip delta
        p_del = (35*m - 5)/(42*(M_inf**2))

        # Temperature at strip delta using state equation
        T_del = gamma*(M_inf**2)*p_del/rho_del

        # Solve theta from stack of 7 Equations
        def solve_theta(theta):
            """
            Frozen shock relation to solve theta

            Args:
                theta: Unkonwn half cone angle to solve [rad]

            Returns:
                eqn25: Equation with only theta as unknown for fsolve

            """
            # Because Numpy has no cot :(
            cot_theta = 1/np.tan(theta)
            cot_beta = 1/np.tan(Beta)
            cot_lambda = 1/np.tan(Beta - theta)

            #  u_del, v_del from frozen shock relations
            u_del = (1-a)*np.cos(theta) + a*np.sin(theta)*cot_beta
            v_del = -(1-a)*np.sin(theta) + a*np.cos(theta)*cot_beta

            # Equation (26)
            u_0 = u_del + (v_del/(2*cot_lambda + cot_theta))

            # Obtain T0 from Energy Equation (5)
            T_0 = 1 + ((gamma-1)/2)*(M_inf**2)*(1 - u_0**2)

            # Equation (23)
            rho_0 = (-rho_del*v_del*(cot_lambda + cot_theta))/u_0

            # Use State Equation to get p0
            p_0 = ((rho_0*T_0)/(gamma*(M_inf**2)))

            # Equation (25)
            eqn25 = 2*(cot_lambda + cot_theta)*(v_del**2)*rho_del - (2*cot_lambda + cot_theta)*(p_0 - p_del)

            return eqn25

        # Perform solving by guessing theta~0.1
        theta = sci.optimize.fsolve(solve_theta, [0.01])[0]

        # Substitue theta back to other unkowns
        cot_theta = 1/np.tan(theta)
        cot_beta = 1/np.tan(Beta)
        cot_lambda = 1/np.tan(Beta - theta)

        #  u_del, v_del from frozen shock relations
        u_del = (1-a)*np.cos(theta) + a*np.sin(theta)*cot_beta
        v_del = -(1-a)*np.sin(theta) + a*np.cos(theta)*cot_beta

        # Equation (26)
        u_0 = u_del + (v_del/(2*cot_lambda + cot_theta))

        # Obtain T0 from Energy Equation (5)
        T_0 = 1 + ((gamma-1)/2)*(M_inf**2)*(1 - u_0**2)

        # Equation (23)
        rho_0 = (-rho_del*v_del*(cot_lambda + cot_theta))/u_0

        # Use State Equation to get p0
        p_0 = ((rho_0*T_0)/(gamma*(M_inf**2)))

        # Equation (25)
        p_del = (2*(cot_lambda + cot_theta)*(v_del**2)*rho_del - (2*cot_lambda + cot_theta)*p_0)/-(2*cot_lambda + cot_theta)    
        
        v_max = (u_del**2 + v_del**2)**0.5

        if full_output == True:
            return [np.rad2deg(theta), u_del, u_0, v_del, p_0, p_del, T_0, T_del, rho_0, rho_del]
        else:
            return np.rad2deg(theta)


    def two_strip_relations(self, unkns):
        
        # Call Class Variables
        M_inf = self.Mach
        gamma = self.gamma
        Beta = self.Beta
        
        # Extract Unknows
        u_0 = unkns[0]
        u_1 = unkns[1]
        v_1 = unkns[2]
        rho_0 = unkns[3]
        rho_1 = unkns[4]
        theta = unkns[5]

        # Because Numpy has no cot :(
        cot_theta = 1/np.tan(theta)
        cot_beta = 1/np.tan(Beta)
        Lambda = Beta - theta
        cot_Lambda = 1/np.tan(Lambda)  

        # Define r_i
        r_0 = 1 
        r_1 = 1 + 0.5*np.tan(Lambda)*cot_theta
        r_del = 1 + np.tan(Lambda)*cot_theta

        # Frozen Shock Relations
        m = (M_inf**2)*(np.sin(Beta)**2)
        a = 5*(m-1)/(6*M_inf**2)

        # Density at strip_delta
        rho_del = (6*m)/(5 + m)

        # Pressure at strip delta
        p_del = (35*m - 5)/(42*(M_inf**2))

        # Temperature at strip delta using state equation
        T_del = gamma*(M_inf**2)*p_del/rho_del

        #  u_del, v_del from frozen shock relations
        u_del = (1-a)*np.cos(theta) + a*cot_beta*np.sin(theta)
        v_del = -(1-a)*np.sin(theta) + a*cot_beta*np.cos(theta)

        # Energy and state equations
        p_0 = (rho_0/(gamma*(M_inf**2)))*(1 + (1 - u_0**2)*((gamma-1)/2)*(M_inf**2))
        p_1 = (rho_1/(gamma*(M_inf**2)))*(1 + (1 - (u_1**2) - (v_1**2))*((gamma-1)/2)*(M_inf**2))

        # Integral Relation Unknows
        # Q Terms
        Q_0 = np.array([(rho_0*u_0)*r_0, (p_0 + rho_0*(u_0**2))*r_0 , 0])
        Q_1 = np.array([(rho_1*u_1)*r_1, (p_1 + rho_1*(u_1**2))*r_1, (rho_1*u_1*v_1)*r_1])
        Q_del = np.array([(rho_del*u_del)*r_del, (p_del + rho_del*(u_del**2))*r_del, (rho_del*u_del*v_del)*r_del])
        # G Terms
        G_0 = np.array([0, 0, p_0*r_0])
        G_1 = np.array([(rho_1*v_1)*r_1, (rho_1*u_1*v_1)*r_1, (p_1 + rho_1*(v_1**2))*r_1])
        G_del = np.array([(rho_del*v_del)*r_del, (rho_del*u_del*v_del)*r_del, (p_del + rho_del*(v_del**2))*r_del])
        # F Terms
        F_0 = np.array([0, p_0, p_0*cot_theta])
        F_1 = np.array([0, p_1, p_1*cot_theta])
        F_del = np.array([0, p_del, p_del*cot_theta])

        # Jerry.S Coefficients
        # # Integrate from 0 -> del
        # I0_C = (Q_0[0] - 4*Q_1[0] + 3*Q_del[0])*np.tan(Lambda) - 4*(G_0[0] - 2*G_1[0] + G_del[0]) - np.tan(Lambda)*(F_0[0] - F_del[0])
        # I0_XM = (Q_0[1] - 4*Q_1[1] + 3*Q_del[1])*np.tan(Lambda) - 4*(G_0[1] - 2*G_1[1] + G_del[1]) - np.tan(Lambda)*(F_0[1] - F_del[1])
        # I0_YM = (Q_0[2] - 4*Q_1[2] + 3*Q_del[2])*np.tan(Lambda) - 4*(G_0[2] - 2*G_1[2] + G_del[2]) - np.tan(Lambda)*(F_0[2] - F_del[2])

        # # Integrate from 0 -> 1/2 del
        # I1_C = 4*(Q_1[0] - Q_del[0])*np.tan(Lambda) - G_0[0] - 4*G_1[0] + 5*G_del[0] - np.tan(Lambda)*(2*F_1[0] + F_del[0])
        # I1_XM = 4*(Q_1[1] - Q_del[1])*np.tan(Lambda) - G_0[1] - 4*G_1[1] + 5*G_del[1] - np.tan(Lambda)*(2*F_1[1] + F_del[1])
        # I1_YM = 4*(Q_1[2] - Q_del[2])*np.tan(Lambda) - G_0[2] - 4*G_1[2] + 5*G_del[2] - np.tan(Lambda)*(2*F_1[2] + F_del[2])

        # Dr.D Coefficients
        # # Integrate from 0 -> 1/2del
        I0_C = (5*Q_0[0] - 4*Q_1[0] - 1*Q_del[0])*np.tan(Lambda) + 24*(G_1[0] - 0*G_0[0]) - (5*F_0[0] + 8*F_1[0] - F_del[0])*np.tan(Lambda)
        I0_XM = (5*Q_0[1] - 4*Q_1[1] - 1*Q_del[1])*np.tan(Lambda) + 24*(G_1[1] - 1*G_0[1]) - (5*F_0[1] + 8*F_1[1] - F_del[1])*np.tan(Lambda)
        I0_YM = (5*Q_0[2] - 4*Q_1[2] - 1*Q_del[2])*np.tan(Lambda) + 24*(G_1[2] - 1*G_0[2]) - (5*F_0[2] + 8*F_1[2] - F_del[2])*np.tan(Lambda)

        # # Integrate from 0 -> del
        I1_C = (1*Q_0[0] + 4*Q_1[0] - 5*Q_del[0])*np.tan(Lambda) + 6*(G_del[0] - 0*G_0[0]) - (1*F_0[0] + 4*F_1[0] + 1*F_del[0])*np.tan(Lambda)
        I1_XM = (1*Q_0[1] + 4*Q_1[1] - 5*Q_del[1])*np.tan(Lambda) + 6*(G_del[1] - 1*G_0[1]) - (1*F_0[1] + 4*F_1[1] + 1*F_del[1])*np.tan(Lambda)
        I1_YM = (1*Q_0[2] + 4*Q_1[2] - 5*Q_del[2])*np.tan(Lambda) + 6*(G_del[2] - 1*G_0[2]) - (1*F_0[2] + 4*F_1[2] + 1*F_del[2])*np.tan(Lambda)

        # # Integrate from 1/2 del -> del
        # I1_C = (-Q_0[0] - 16*Q_1[0] + 5*Q_del[0])*np.tan(Lambda) + 24*(G_del[0] - G_1[0]) - (-F_0[0] + 8*F_1[0] + 5*F_del[0])*np.tan(Lambda)
        # I1_XM = (-Q_0[1] - 16*Q_1[1] + 5*Q_del[1])*np.tan(Lambda) + 24*(G_del[1] - G_1[1]) - (-F_0[1] + 8*F_1[1] + 5*F_del[1])*np.tan(Lambda)
        # I1_YM = (-Q_0[2] - 16*Q_1[2] + 5*Q_del[2])*np.tan(Lambda) + 24*(G_del[2] - G_1[2]) - (-F_0[2] + 8*F_1[2] + 5*F_del[2])*np.tan(Lambda)

        eqns_to_solve = np.array([I0_C, I0_XM, I0_YM, I1_C, I1_XM, I1_YM])

        print(unkns, eqns_to_solve)

        return eqns_to_solve


    def two_strip_solve(self, full_output:bool):

        one_strip_sol = self.one_strip_solve(full_output=True)

        # These can be grab from 1 strip
        u_del =  one_strip_sol[1]
        u_0 =  0.7913637553006431   #one_strip_sol[2]
        u_1 =  0.7919718861644549   #(u_0 + u_del)/2
        v_del =  one_strip_sol[3]
        v_1 = -0.030536066612168594   #v_del/2
        rho_0 = 5.192282199293422    #one_strip_sol[8]
        rho_del = one_strip_sol[9]
        rho_1 = 5.138006805966549   #(rho_0 + rho_del)/2
        theta = np.deg2rad(33.535712892301554)  #one_strip_sol[0])

        # Wrap Initial guess
        initial_guess =  np.array([u_0, u_1, v_1, rho_0, rho_1, theta])

        solutions = sci.optimize.fsolve(self.two_strip_relations, initial_guess, full_output=True)
        
        if solutions[3] == 'The solution converged.':
            solver_msg = 'Converged.'
        else:
            solver_msg = 'Not converged.'
        # print('theta N=2: ' + str(np.rad2deg(solutions[0][-1])))
        
        u_0 = solutions[0][0]
        u_1 = solutions[0][1]
        v_1 = solutions[0][2]
        rho_0 = solutions[0][3]
        rho_1 = solutions[0][4]
        theta = solutions[0][5]

        print(np.rad2deg(theta))

        if full_output == True:
            return np.rad2deg(theta), u_0, u_1, v_1, rho_0, rho_1, solver_msg
        else:
            return np.rad2deg(theta), solver_msg


    def main():
        #  1-Strip - Testing
        print(' ')

        global M_inf
        global Beta
        global gamma

        M_inf = 8
        Beta = 38.155
        Beta = np.deg2rad(Beta)
        gamma = 1.4

        # 1-Strip Solving
        [Theta, u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0] = frozen_one_strip_cone(M_inf, Beta, gamma)

        print('1-Strip Conical Frozen Flow\n')
        print([np.rad2deg(Theta), u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0])
        print('\n')

        # 2-Strip Solving
        # Inital Guess   | M = 1.5 Beta = 60 | M = 8 Beta = 38.155
        u_0 =  0.7915200156896637 #0.6684754503861963 #
        u_1 =  0.7919718861644587 #0.6846846749618256 #
        v_1 = -0.030536066611886743 #-0.12949087134420384 #
        rho_0 = 5.192282199293422 #1.707469061399742 #
        rho_1 = 5.138006805966865 #1.6485648593002313 #
        theta = 0.5847021888888834 #0.32535828738426736 #

        initial_guess =  np.array([u_0, u_1, v_1, rho_0, rho_1, theta])

        solution = frozen_two_strip_cone_solve(initial_guess)
        print(solution)

        return


class Frozen_Blunt_Flow:

    def __init__(self, M_inf, gamma, N):

        self.M_inf = M_inf
        self.gamma = gamma
        self.N = N

        # Freestream dimensionless Speed of Sound
        self.c_inf = np.sqrt(((gamma-1)/2)/(1 + ((gamma-1)/2)*(M_inf**2)))

        # Freestream dimensionless speed
        self.w_inf = M_inf*self.c_inf

        # A constant from Dr.B
        self.k = (gamma-1)/(2*gamma)

        self.E0_list = []
        self.theta_list = []
    

    def update_unknown(self,theta, epsilon, sigma, v_0):


        self.v_0 = v_0
        self.sigma = sigma
        self.epsilon = epsilon
        
        self.theta_list.append(theta)
        self.E0_list.append(self.E(0, theta))
        
        return
    

    def epsilon_0(self):

        # Interp from Dr.B Fig 4
        Mach_inf = [2.157664347,2.381765785,2.5105146,2.86600679,2.997974516,3.501764327,4.008773049,4.502184911,5.005724194]

        eps_0 = [1.2470301,0.996510823,0.914273741,0.745007653,0.698172509,0.606834414,0.550898258,0.513464762,0.477743969]

        eps_0_interped = np.interp(self.M_inf, Mach_inf, eps_0)

        return eps_0_interped

    
    def w_x(self, sigma: float) -> float:     
        """ x component of w behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:

        Raises:

        Returns:
            w_x {float} -- w_x as a function of sigma
        """
        # self.sigma = sigma

        w_x = self.w_inf*(1 - (2/(self.gamma + 1))*(np.sin(self.sigma)**2)*(1 - 1/((self.M_inf**2)*(np.sin(self.sigma)**2))))

        return w_x


    def w_y(self, sigma: float) -> float:     
        """ y component of w behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:

        Raises:

        Returns:
            w_y {float} -- w_y as a function of sigma
        """
        # Store to class variable so can call in u, v function
        # self.sigma = sigma

        w_y = (self.w_inf/(self.gamma + 1))*np.sin(2*self.sigma)*(1 - (1/((self.M_inf**2)*(np.sin(self.sigma)**2))))

        return w_y

    
    def u(self, index: int, theta: float=0) -> float:
        """ r component of w behind shock. 
            u=0 on cylinder surface.

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]. 0 if not specified.

            \u03B8 {float} -- Polar coordinate system independent variable [rad]. 0 if not specified.

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermidiate strips
        Raises:

        Returns:
            u {float} -- u as a function of \u03B8 [rad] at i
        """
        
        if index == 0:
            return 0
        if index == 1:
            return self.w_y(self.sigma)*np.sin(theta) - self.w_x(self.sigma)*np.cos(theta)
        else:
            exit(f"u_{str(index)} is not a known Boundary Condition.")
    

    def v(self, index: int, theta: float=0) -> float:
        """ \u03B8 component of w behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]. 0 if not specified.

            \u03B8 {float} -- Polar coordinate system independent variable [rad]. 0 if not specified.

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermidiate strips

        Raises:
            SystemExit(Not a known Boundary Condition)
        Returns:
            v {float} -- v as a function of \u03B8 [rad]
        """

        if index == 0 and theta == 0:
            return 0
        if index == 1:
            return self.w_x(self.sigma)*np.sin(theta) + self.w_y(self.sigma)*np.cos(theta)
        else:
            return self.v_0
            # exit(f"v_{str(index)} is not a known Boundary Condition.")

    
    def w(self, index, theta):
        """ Total velocity

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]. 0 if not specified.

            \u03B8 {float} -- Polar coordinate system independent variable [rad]. 0 if not specified.

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermidiate strips

        Raises:
            SystemExit(Not a known Boundary Condition)
        Returns:
            w {float} -- w as a function of \u03B8 [rad]
        """

        u = self.u(index,theta)
        v = self.v(index,theta)

        return np.sqrt(u**2 + v**2)

    
    def tau(self, index, theta):

        return (1 - self.w(index,theta)**2)**(1/(self.gamma-1))

    
    def c(self, index, theta):

        return np.sqrt(((self.gamma-1)/2)*(1 - self.w(index, theta)**2))
    

    def M(self, index, theta):

        return -self.t(index, theta)*self.u(index, theta)/self.c(index, theta)**2


    def r(self, index):

        xi = (self.N - index + 1)/self.N

        return 1 + xi*self.epsilon


    def p(self, index: int, theta: float) -> float:

        """ Pressure immediatly behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermidiate strips

        Raises:

        Returns:
            p_1 {float} -- p_1 as a function of \u03C3 [rad]
        """

        gamma = self.gamma
        w_inf = self.w_inf

        if index == 1:
            return ((4*gamma)/((gamma**2)-1))*((1 - w_inf**2)**(gamma/(gamma-1)))*(((w_inf**2*(np.sin(self.sigma)**2))/(1 - w_inf**2)) - (gamma - 1)**2/(4*gamma))
        
        else:
            return self.tau(index, theta)*self.phi(index)**(-1/(self.gamma-1))


    def rho(self, index: int, theta: float) -> float:

        """ Density immediatly behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:

        Raises:

        Returns:
            \u03C1_1 {float} -- \u03C1_1 as a function of \u03C3 [rad]
        """

        gamma = self.gamma
        w_inf = self.w_inf

        if index == 1:
            return ((gamma+1)/(gamma-1))*((1 - w_inf**2)**(1/(gamma-1)))*((w_inf**2)/(1 + (1 - w_inf**2)*(1/np.tan(self.sigma))**2))   
        else:
            return (self.tau(index, theta)**(self.gamma))*self.phi(index)**(-1/(self.gamma-1))


    def phi(self, index) -> float:
        """ Vorticity function immediatly behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:

        Raises:

        Returns:
            \u03C6_1 {float} -- \u03C6 = p/\u03C1^\u03B3
        """

        gamma = self.gamma
        w_inf = self.w_inf

        omega = (w_inf**2*(np.sin(self.sigma)**2))/(1 - w_inf**2)
        
        if index == 0:
            return ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(((w_inf**2)/(1 - w_inf**2)) - (((gamma - 1)**2)/(4*gamma)))*(1/(w_inf**(2*gamma)))

        # phi_i = phi_1 where i = 1,2,3...
        else:
            return ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(omega - (((gamma-1)**2)/(4*gamma)))*(1 + (1/omega))**gamma


    def psi(self, index, theta: float, epsilon: float) -> float:
        """ Stream function without discontinuity

        Arguments:
            \u03B8 {float} -- Polar coordinate system independent variable [rad]
            \u03B5 {float} -- Shock displacement distance

        Keyword Arguments:

        Raises:

        Returns:
            \u03C8 {float} -- \u03C8_{i} where \u03C8_{i} = \u03C8_{1}
        """

        gamma = self.gamma
        w_inf = self.w_inf

        if index == 0:
            return 0

        else:
            return w_inf*(1 - w_inf**2)**(1/(gamma-1))*(1 + epsilon)*np.sin(theta)

    ######                  Derivative Terms                        ######
    
    def deps_dtheta(self, theta: float):
        """ d\u03B5/d\u03B8 :Change of shock wave distance as derivative of theta

        Arguments:
            \u03B8 {float} -- Polar coordinate system independent variable [rad]
            
            \u03B5 {float} -- Shock displacement distance

        Keyword Arguments:

        Raises:

        Returns:
            d\u03B5/d\u03B8 {float}
        """

        return -(1 + self.epsilon)*(1/np.tan(self.sigma + theta))
    

    def dsigma_dtheta(self, theta:float):

        return self.F(theta)
    

    def du_dtheta(self, index, theta):

        if index == 0:
            return 0
        
        else:
            t_i = self.t(index, theta)
            t_i_prime = self.t_prime(index, theta)
            phi_i = self.phi(index, theta)
            return "du_i/dtheta: Not done yet. 1-strip no need"


    def dv_dtheta(self, index, theta):

        gamma = self.gamma
        
        if index == 0:
            return self.E(0, theta)/(((gamma-1)/(gamma+1)) - self.w(0,theta)**2)
        
        else:
            return self.E(0, theta)/(((gamma-1+2*(self.u(index, theta)**2))/(gamma+1)) - self.w(index,theta)**2)


    def dpsi_dtheta(self, index, theta):

        return print("dpsi_i/dtheta: Not done yet. 1-strip no need")

    ######                  Grouped Terms                        ######
    
    def F(self, theta):

        s_prime1 = self.s_prime(1, theta)
        rho1 = self.rho(1, theta)
        u1 = self.u(1, theta)
        v1 = self.v(1, theta)

        return (s_prime1 - rho1*(v1**2 - u1**2))/self.D_1(theta)

    
    def D_1(self, theta):

        m_1 = self.m_1(theta)
        n_1 = self.n_1(theta)
        u_1 = self.u(1, theta)
        v_1 = self.v(1, theta)
        w_1 = self.w(1, theta)
        gamma = self.gamma
        sigma = self.sigma
        w_inf = self.w_inf
        rho_1 = self.rho(1, theta)

        return (4*gamma/((gamma**2)-1))*w_inf**2*(1 - w_inf**2)**(1/(gamma-1))*(u_1*v_1*np.sin(sigma))/(1 - w_1**2) + rho_1*(v_1*m_1 - u_1*(n_1 + (2*v_1/(1 - w_1**2))*(v_1*n_1 - u_1*m_1)))


    def m_1(self, theta):

        dwydsigma = (2*self.w_inf/(self.gamma + 1))*(np.cos(2*self.sigma) + (((1/np.sin(self.sigma))**2)/(self.M_inf**2)))
        dwxdsigma = (-2*self.w_inf/(self.gamma + 1))*np.sin(2*self.sigma)

        return dwydsigma*np.sin(theta) - dwxdsigma*np.cos(theta)
    

    def n_1(self, theta):

        dwydsigma = (2*self.w_inf/(self.gamma + 1))*(np.cos(2*self.sigma) + (((1/np.sin(self.sigma))**2)/(self.M_inf**2)))
        dwxdsigma = (-2*self.w_inf/(self.gamma + 1))*np.sin(2*self.sigma)

        return -dwxdsigma*np.sin(theta) - dwydsigma*np.cos(theta)
    

    def G_1(self, theta):

        v_1 = self.v(1, theta)
        u_1 = self.u(1, theta)
        c_1 = self.c(1, theta)
        n_1 = self.n_1(theta)
        m_1 = self.m_1(theta)

        return self.tau(1,theta)*((v_1/(c_1**2))*(v_1*n_1 - u_1*m_1) - n_1)
    

    def E(self, index, theta):

        if index == 0:
            return (2*(self.c(index, theta)**2)/((self.gamma + 1)*self.tau(index, theta)))*self.t_prime(index, theta)
    
        else:
            return (2*(self.c(index, theta)**2)/((self.gamma + 1)*self.tau(index, theta)))*(self.t_prime(index, theta) - self.M(index, theta)*self.du_dtheta(index, theta))


    def s(self, index, theta):

        return self.rho(index, theta)*self.u(index, theta)*self.v(index, theta)


    def t(self, index, theta):

        return self.tau(index, theta)*self.v(index, theta)

    ###             MIR Relations           ###

    def s_prime(self, index, theta):

        H = lambda ind: self.k*self.p(ind, theta) + self.rho(ind, theta)*(self.u(ind, theta)**2)
        g = lambda ind: self.k*self.p(ind, theta) + self.rho(ind, theta)*(self.v(ind, theta)**2)

        if index == 1:
            return (1/(self.epsilon))*(self.s(1, theta)*self.deps_dtheta(theta) + 2*H(0) - 2*(1 + self.epsilon)*H(1)) + g(0) + g(1)
        elif index == 0:
            return 0       
        else:
            return print("1-strip method no need s'(i), i=2,3,... ")
    

    def t_prime(self, index, theta):
        
        h = lambda ind: self.tau(ind, theta)*self.u(ind, theta)

        if index == 0:
            return (1/(self.epsilon))*(self.t(1, theta) - self.t(0, theta))*self.deps_dtheta(theta) - self.G_1(theta)*self.dsigma_dtheta(theta) - (1 + (2/self.epsilon))*h(1)
        else:
            return print("1-strip method no need t'(i), i=1,2,3,... ")

    ###         Slover Functions            ###

    def One_Strip_Sys(self, theta, unks: list):

        epsilon = unks[0]
        sigma = unks[1] 
        v_0 = unks[2]  

        # Check Singular
        if v_0**2 <= (self.gamma - 1)/(self.gamma + 1):

            self.update_unknown(theta, epsilon, sigma, v_0)

            desp_dtheta = self.deps_dtheta(theta)
            dsig_dtheta = self.dsigma_dtheta(theta)
            dv0_dtheta = self.dv_dtheta(0, theta)

            return [desp_dtheta, dsig_dtheta, dv0_dtheta]
        else:

            return [0, 0, 0]


    def One_Strip_Solve(self):

        # Start Stop Theta [0 -> theta^(0): 0.8125]
        theta_range = [0, 0.812]

        # Initial Guess
        esp_0 = 0.703 #self.epsilon_0()
        sigma_0 = np.pi/2
        v_0 = self.v(0, theta_range[0])

        IGs = [esp_0, sigma_0, v_0]

        # Run Solver
        Flow_Solution = sci.integrate.solve_ivp(self.One_Strip_Sys, theta_range, IGs, max_step=0.001) 

        return Flow_Solution
