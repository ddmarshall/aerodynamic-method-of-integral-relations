import sys
import numpy as np
import scipy as sci
import json as js
# from scipy import optimize
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Darwin':
    plt.switch_backend('MacOSX')
import warnings
warnings.filterwarnings("error")


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

    def __init__(self, M_inf, gamma, N, cutoff):
        print('\nPython: ' + sys.version)
        print('Scipy Version: ' + sci.__version__)
        print('Numpy Version: ' + np.__version__)
        print('Method of Integral Relations')
    
        self.M_inf = M_inf
        self.gamma = gamma
        self.N = N
        self.cutoff = cutoff

        # Freestream dimensionless Speed of Sound
        self.c_inf = np.sqrt(((gamma-1)/2)/(1 + ((gamma-1)/2)*(M_inf**2)))

        # Freestream dimensionless speed
        self.w_inf = M_inf*self.c_inf

        # A constant from Dr.B
        self.k = (gamma-1)/(2*gamma)    
        print(f'{self.N}-Strip Cylinder Solution \nMach: {self.M_inf:3.2f} Gamma: {self.gamma:3.2f}\nODE cutoff at {cutoff*100}% \n')
        

    def epsilon_0(self):

        # Interp from Dr.B Fig 4 (for 1-D interp)
        Mach_inf = [2.157664347,2.381765785,2.5105146,2.86600679,2.997974516,3.501764327,4.008773049,4.502184911,5.005724194]

        eps_0 = [1.2470301,0.996510823,0.914273741,0.745007653,0.698172509,0.606834414,0.550898258,0.513464762,0.477743969]
        '''
        # Mach_inf = [2.164985590778098,2.5144092219020173,3.0043227665706054,3.5050432276657064,4.012968299711816,4.502881844380404,5.003602305475505] 
        
        # eps_0 = [1.25, 0.9136690647482014, 0.6960431654676258, 0.6025179856115107, 0.5467625899280575, 0.5053956834532372, 0.46942446043165464]


        # use 2nd order
        # p_coeff = np.polyfit(Mach_inf, eps_0, 2)
        # eps_0_interped = np.polyval(p_coeff, self.M_inf)
        
        # plt.figure()
        # plt.scatter(Mach_inf, eps_0)
        # plt.plot(Mach_inf, np.polyval(p_coeff, Mach_inf))
        '''
        # eps_0_interped = np.interp(self.M_inf, Mach_inf, eps_0)

        # Fit with exponential model
        def func(x, a, c, d):
            return a*np.exp(-c*x)+d

        popt, pcov = sci.optimize.curve_fit(func, Mach_inf, eps_0, p0=(1, 1e-6, 1))
        eps_0_interped = func(self.M_inf, *popt)

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

        w_x = self.w_inf*(1 - (2/(self.gamma + 1))*(np.sin(sigma)**2)*(1 - 1/((self.M_inf**2)*(np.sin(sigma)**2))))

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

        w_y = (self.w_inf/(self.gamma + 1))*np.sin(2*sigma)*(1 - (1/((self.M_inf**2)*(np.sin(sigma)**2))))

        return w_y

    
    def u(self, index: int, theta: float, sigma: float) -> float:
        """ r component of w behind shock. 
            u=0 on cylinder surface.

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]. 0 if not specified.

            \u03B8 {float} -- Polar coordinate system independent variable [rad]. 0 if not specified.

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermediate strips
        Raises:

        Returns:
            u {float} -- u as a function of \u03B8 [rad] at i
        """
        
        if index == 0:
            return 0
        elif index == 1:
            return self.w_y(sigma)*np.sin(theta) - self.w_x(sigma)*np.cos(theta)
        elif index == 2:
            # Need to add u_2 as a function
            return self.u_2(theta)
        else:
            exit(f"u_{str(index)} is not a known Boundary Condition.")
    

    def v(self, index: int, theta: float, sigma: float) -> float:
        """ \u03B8 component of w behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]. 0 if not specified.

            \u03B8 {float} -- Polar coordinate system independent variable [rad]. 0 if not specified.

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermediate strips

        Raises:
            SystemExit(Not a known Boundary Condition)
        Returns:
            v {float} -- v as a function of \u03B8 [rad]
        """

        # if index == 0 and theta == 0:
        #     return 0
        if index == 1:
            return self.w_x(sigma)*np.sin(theta) + self.w_y(sigma)*np.cos(theta)
        elif index == 0:
            # Return surface v
            return self.v_0(theta)
        elif index == 2:
            # Return surface v
            if self.v_2(theta) > 1:
                print(self.v_2(theta))
            return self.v_2(theta)
        else:
            exit(f"v_{str(index)} is not a known Boundary Condition.")

    
    def w(self, index, theta, sigma):
        """ Total velocity

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]. 0 if not specified.

            \u03B8 {float} -- Polar coordinate system independent variable [rad]. 0 if not specified.

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermediate strips

        Raises:
            SystemExit(Not a known Boundary Condition)
        Returns:
            w {float} -- w as a function of \u03B8 [rad]
        """

        u = self.u(index, theta, sigma)
        v = self.v(index, theta, sigma)

        return np.sqrt(u**2 + v**2)

    
    def tau(self, index, theta, sigma):
        try:
            tau = (1 - self.w(index, theta, sigma)**2)**(1/(self.gamma-1))
        except RuntimeWarning:
            tau = float('NaN')
        return tau

    
    def c(self, index, theta, sigma):
        try:
            c = np.sqrt(((self.gamma-1)/2)*(1 - self.w(index, theta, sigma)**2))
        except RuntimeWarning:
            c = float('NaN')
        return c
    

    def M(self, index, theta, sigma):

        return -self.t(index, theta, sigma)*self.u(index, theta, sigma)/self.c(index, theta, sigma)**2


    def r(self, index, epsilon):

        xi = (self.N - index + 1)/self.N

        if index == 0:
            return np.ones(len(epsilon))
        else:
            return 1 + xi*epsilon


    def p(self, index: int, theta: float, sigma: float, epsilon) -> float:

        """ Pressure immediatly behind shock

        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:
            index {int}
                    -- 0: Surface
                    -- 1: Wave
                    -- i: Intermediate strips

        Raises:

        Returns:
            p_1 {float} -- p_1 as a function of \u03C3 [rad]
        """

        gamma = self.gamma
        w_inf = self.w_inf

        omega = ((w_inf**2)*(np.sin(sigma)**2))/(1 - w_inf**2)
        phi_1 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(omega - (((gamma-1)**2)/(4*gamma)))*(1 + (1/omega))**gamma

        if index == 1:
            return ((4*gamma)/((gamma**2)-1))*((1 - w_inf**2)**(gamma/(gamma-1)))*(((w_inf**2*(np.sin(sigma)**2))/(1 - w_inf**2)) - (((gamma - 1)**2)/(4*gamma)))
        
        else:
            if theta == 0 and index == 2:
                return (self.tau(2, theta, sigma)**gamma)*(phi_1**(-1/(gamma-1)))
            else:
                return (self.tau(index, theta, sigma)**gamma)*self.phi(index, theta, sigma, epsilon)**(-1/(gamma-1))


    def rho(self, index: int, theta: float, sigma: float, epsilon) -> float:

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

        omega = ((w_inf**2)*(np.sin(sigma)**2))/(1 - w_inf**2)
        phi_1 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(omega - (((gamma-1)**2)/(4*gamma)))*(1 + (1/omega))**gamma

        if index == 1:
            return ((gamma+1)/(gamma-1))*((1 - w_inf**2)**(1/(gamma-1)))*((w_inf**2)/(1 + (1 - w_inf**2)*(1/np.tan(sigma))**2))   
        else:
            if theta == 0 and index == 2:
                return self.tau(2, theta, sigma)*(phi_1**(-1/(gamma-1)))
            else:
                return self.tau(index, theta, sigma)*(self.phi(index, theta, sigma, epsilon)**(-1/(gamma-1)))


    def phi(self, index, theta, sigma, epsilon) -> float:
        """ Vorticity function immediatly behind shock
            Contant and equal to shock properties
        Arguments:
            \u03C3 {float} -- Inclination of shock wave to the direction of incident stream line [rad]

        Keyword Arguments:

        Raises:

        Returns:
            \u03C6_1 {float} -- \u03C6 = p/\u03C1^\u03B3
        """

        gamma = self.gamma
        w_inf = self.w_inf

        omega = ((w_inf**2)*(np.sin(sigma)**2))/(1 - w_inf**2)

        phi_0 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(((w_inf**2)/(1 - w_inf**2)) - (((gamma - 1)**2)/(4*gamma)))*(1/(w_inf**(2*gamma)))

        phi_1 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(omega - (((gamma-1)**2)/(4*gamma)))*(1 + (1/omega))**gamma

        if index == 0:
            return phi_0

        elif index == 1:
            return phi_1

        elif index == 2:
            if theta > 0:

                dphi0_dpsi0 = 0

                phi_i, dphi_dpsi_i = self.Two_Strip_Interp(x=[self.psi(0, theta, epsilon),self.psi(1, theta, epsilon)],y=[phi_0, phi_1],dydx=[dphi0_dpsi0, self.dphi1_dpsi1])

                phi_2 = float(phi_i(self.psi(index, theta, epsilon)))

                # print(phi_0, phi_2, phi_1)
                return phi_2
            else:
                return self.phi2_0


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

        elif index == 1:
            return w_inf*(1 - w_inf**2)**(1/(gamma-1))*(1 + epsilon)*np.sin(theta)
        elif index == 2:
            return self.psi_2(theta)

    ######                  Derivative Terms                        ######
    
    def deps_dtheta(self, theta: float, sigma: float, epsilon : float):
        """ d\u03B5/d\u03B8 :Change of shock wave distance as derivative of theta

        Arguments:
            \u03B8 {float} -- Polar coordinate system independent variable [rad]
            
            \u03B5 {float} -- Shock displacement distance

        Keyword Arguments:

        Raises:

        Returns:
            d\u03B5/d\u03B8 {float}
        """

        return -(1 + epsilon)*(1/np.tan(sigma + theta))
    

    def dsigma_dtheta(self, theta:float, sigma, epsilon):

        return self.F(theta, sigma, epsilon)
    

    def du_dtheta(self, index, theta, sigma, epsilon):

        if index == 0:
            return 0

        elif index == 2:
            s_i = self.s(index, theta, sigma, epsilon)
            t_i = self.t(index, theta, sigma)
            s_i_prime = self.s_prime(index, theta, sigma, epsilon)
            t_i_prime = self.t_prime(index, theta, sigma, epsilon)
            u_i = self.u(index, theta, sigma)
            phi_i = self.phi(index, theta, sigma, epsilon)
            dpsi_dtheta = self.dpsi_dtheta(index, theta, sigma, epsilon)
            gamma = self.gamma    

            '''
            # Omega, dOmega/dSigma, d2Omega/dSigma2
            # omega = ((self.w_inf**2)*(np.sin(sigma)**2))/(1 - self.w_inf**2)
            # domega_dsigma = ((self.w_inf**2)*(2*np.sin(sigma)*np.cos(sigma)))/(1 - self.w_inf**2)
            # d2omega_dsigma2 = 2*(self.w_inf**2)*np.cos(2*sigma)/(1 - self.w_inf**2)

            # dphi1/dSigma, d2phi1/dSigma2
            # dphi0_dsigma = 0

            # dphi1_dsigma = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*((1 - (gamma/(omega+1)))*domega_dsigma*((1+(1/omega))**gamma) - (((gamma-1)**2)/(4*gamma))*-gamma*(1/omega**2)*((1+(1/omega))**(gamma-1))*domega_dsigma)

            # d2phi0_dsigma2 = 0

            # d2phi1_dsigma2 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*((((1+(1/omega))**gamma)*((gamma/(1+(1/omega)))*(-(omega**-2))*(domega_dsigma**2)+d2omega_dsigma2)-gamma*((1+(1/omega))**(gamma-1))*((-omega**-2)*(domega_dsigma**2)-(omega**-3)*((gamma-1)/(1+(1/omega)))*(domega_dsigma**2) + (omega**-1)*d2omega_dsigma2)) - ((((gamma-1)**2)/(4*gamma))*(-gamma*((1+(1/omega))**(gamma-1))*(-2*(omega**-3)*(domega_dsigma**2)-(omega**-4)*((gamma-1)/(1+(1/omega)))*(domega_dsigma**2)+(omega**-2)*((1+(1/omega))**(gamma-1))*d2omega_dsigma2))))

            # dlogphi1_dpsi1 = (1/self.phi(1, theta, sigma, epsilon))*dphi1_dsigma*self.dsigma_dtheta(theta, sigma, epsilon)/self.dpsi_dtheta(1, theta, sigma, epsilon)
            '''
            dphi0_dpsi0 = 0

            if theta > 0:
                phi_i, dphi_dpsi_i = self.Two_Strip_Interp(x=[self.psi(0, theta, epsilon),self.psi(1, theta, epsilon)],y=[self.phi(0, theta, sigma, epsilon), self.phi(1, theta, sigma, epsilon)],dydx=[dphi0_dpsi0, self.dphi1_dpsi1])

                phi_i = phi_i(self.psi(index, theta, epsilon))

                dlogphi_dpsi = (1/phi_i)*dphi_dpsi_i(self.psi(index, theta, epsilon))

            else:
                dlogphi_dpsi = 0

            du_dtheta = (s_i_prime - (u_i*(phi_i**(-1/(gamma-1)))*t_i_prime) + (s_i/(gamma-1))*dlogphi_dpsi*dpsi_dtheta)/(t_i*(phi_i**(-1/(gamma-1))))

            return du_dtheta


    def dv_dtheta(self, index, theta, sigma, epsilon):

        gamma = self.gamma
        
        if index == 0:
            return self.E(index, theta, sigma, epsilon)/(((gamma-1)/(gamma+1)) - self.w(0,theta, sigma)**2)
        
        else:
            return self.E(index, theta, sigma, epsilon)/(((gamma-1+(2*(self.u(index, theta, sigma)**2)))/(gamma+1)) - (self.w(index, theta, sigma)**2))


    def dpsi_dtheta(self, index, theta, sigma, epsilon):
        # i = 2...N

        xi = (self.N - index + 1)/self.N
        dri_dtheta = xi*self.deps_dtheta(theta, sigma, epsilon)

        if index == 1:
            return self.w_inf*((1 - self.w_inf**2)**(1/(self.gamma-1)))*(np.cos(theta) + self.deps_dtheta(theta, sigma, epsilon)*np.sin(theta) + epsilon*np.cos(theta))
        else:
            return self.rho(index, theta, sigma, epsilon)*((dri_dtheta*self.v(index, theta, sigma)) - (self.r(index,epsilon)*self.u(index,theta,sigma)))

    ######                  Grouped Terms                        ######
    
    def F(self, theta, sigma, epsilon):

        s_prime1 = self.s_prime(1, theta, sigma, epsilon)
        rho1 = self.rho(1, theta, sigma, epsilon)
        u1 = self.u(1, theta, sigma)
        v1 = self.v(1, theta, sigma)

        return (s_prime1 - rho1*(v1**2 - u1**2))/self.D_1(theta, sigma, epsilon)

    
    def D_1(self, theta, sigma, epsilon):

        m_1 = self.m_1(theta, sigma)
        n_1 = self.n_1(theta, sigma)
        u_1 = self.u(1, theta, sigma)
        v_1 = self.v(1, theta, sigma)
        w_1 = self.w(1, theta, sigma)
        gamma = self.gamma
        w_inf = self.w_inf
        rho_1 = self.rho(1, theta, sigma, epsilon)

        return (4*gamma/((gamma**2)-1))*w_inf**2*(1 - w_inf**2)**(1/(gamma-1))*(u_1*v_1*np.sin(sigma))/(1 - w_1**2) + rho_1*(v_1*m_1 - u_1*(n_1 + (2*v_1/(1 - w_1**2))*(v_1*n_1 - u_1*m_1)))


    def m_1(self, theta, sigma):

        dwydsigma = (2*self.w_inf/(self.gamma + 1))*(np.cos(2*sigma) + (((1/np.sin(sigma))**2)/(self.M_inf**2)))
        dwxdsigma = (-2*self.w_inf/(self.gamma + 1))*np.sin(2*sigma)

        return dwydsigma*np.sin(theta) - dwxdsigma*np.cos(theta)
    

    def n_1(self, theta, sigma):

        dwydsigma = (2*self.w_inf/(self.gamma + 1))*(np.cos(2*sigma) + (((1/np.sin(sigma))**2)/(self.M_inf**2)))
        dwxdsigma = (-2*self.w_inf/(self.gamma + 1))*np.sin(2*sigma)

        return -dwxdsigma*np.sin(theta) - dwydsigma*np.cos(theta)
    

    def G_1(self, theta, sigma):

        v_1 = self.v(1, theta, sigma)
        u_1 = self.u(1, theta, sigma)
        c_1 = self.c(1, theta, sigma)
        n_1 = self.n_1(theta, sigma)
        m_1 = self.m_1(theta, sigma)

        return self.tau(1,theta, sigma)*((v_1/(c_1**2))*(v_1*n_1 - u_1*m_1) - n_1)
    

    def E(self, index, theta, sigma, epsilon):

        if index == 0:
            
            return (2*(self.c(index, theta, sigma)**2)/((self.gamma + 1)*self.tau(index, theta, sigma)))*self.t_prime(index, theta, sigma, epsilon)
    
        else:
            return (2*(self.c(index, theta, sigma)**2)/((self.gamma + 1)*self.t(index, theta, sigma)))*(self.t_prime(index, theta, sigma, epsilon) - self.M(index, theta, sigma)*self.du_dtheta(index, theta, sigma, epsilon))


    def s(self, index, theta, sigma, epsilon):

        # if self.N == 2:
        #     if index == 1:
        #         return self.s1(theta)
        #     if index == 2:
        #         return self.s2(theta)
        # else:
        return self.rho(index, theta, sigma, epsilon)*self.u(index, theta, sigma)*self.v(index, theta, sigma)


    def t(self, index, theta, sigma):
        # if self.N == 2:
        #     if index == 0:
        #         return self.t0(theta)
            # if index == 1:
                # return self.t1(theta)
            # if index == 2:
                # return self.t2(theta)
            # else:
        return self.tau(index, theta, sigma)*self.v(index, theta, sigma)

    ###             MIR Relations           ###

    def s_prime(self, index, theta, sigma, epsilon):

        H = lambda ind: self.k*self.p(ind, theta, sigma, epsilon) + self.rho(ind, theta, sigma, epsilon)*(self.u(ind, theta, sigma)**2)

        g = lambda ind: self.k*self.p(ind, theta, sigma, epsilon) + self.rho(ind, theta, sigma, epsilon)*(self.v(ind, theta, sigma)**2)

        s1 = self.s(1, theta, sigma, epsilon)
        
        if self.N == 1:
            if index == 1:
                return (1/(epsilon))*(self.s(1, theta, sigma, epsilon)*self.deps_dtheta(theta, sigma, epsilon) + 2*H(0) - 2*(1 + epsilon)*H(1)) + g(0) + g(1)
            elif index == 0:
                return 0       
            else:
                return print("1-strip method no need s'(i), i=2,3,... ")

        elif self.N == 2:
            s2 = self.s(2, theta, sigma, epsilon)
            if index == 1:
                return (1/(epsilon))*((3*s1 - 4*s2)*self.deps_dtheta(theta, sigma, epsilon) - 4*(H(0) + (1+epsilon)*H(1) - (2+epsilon)*H(2))) + g(1) - g(0)

            elif index == 2:
                return (1/(2*epsilon))*(s1*self.deps_dtheta(theta, sigma, epsilon) + 5*H(0) - (1+epsilon)*H(1) - 2*(2+epsilon)*H(2)) + (g(0)/2) + g(2)
 

    def t_prime(self, index, theta, sigma, epsilon):
        
        h = lambda ind: self.tau(ind, theta, sigma)*self.u(ind, theta, sigma)

        if self.N == 1:
            if index == 0:
                return (1/epsilon)*(self.t(1, theta, sigma) - self.t(0, theta, sigma))*self.deps_dtheta(theta, sigma, epsilon) - self.G_1(theta, sigma)*self.dsigma_dtheta(theta, sigma, epsilon) - (1 + (2/epsilon))*h(1)
            else:
                return print("1-strip method no need t'(i), i=1,2,3,... ")
        
        elif self.N == 2:

            t_1_prime = self.G_1(theta, sigma)*self.dsigma_dtheta(theta, sigma, epsilon) - h(1)
            t0 = self.t(0, theta, sigma)
            t1 = self.t(1, theta, sigma)
            t2 = self.t(2, theta, sigma)

            if index == 0:
                return t_1_prime + (1/epsilon)*(4*((1+epsilon)*h(1) - (2+epsilon)*h(2)) - (t0 + 3*t1 - 4*t2)*self.deps_dtheta(theta, sigma, epsilon))
            elif index == 1:
                return t_1_prime
            elif index == 2:
                return (-t_1_prime/2) + (1/epsilon)*(2*(t1 - t2)*self.deps_dtheta(theta, sigma, epsilon) + (2+epsilon)*h(2) - 2.5*(1+epsilon)*h(1))


    ###         Slover Functions            ###

    def One_Strip_Sys(self, theta, unks: list):

        epsilon = unks[0]
        sigma = unks[1] 
        v_0 = unks[2]  
            
        # Keep format consistent
        self.v_0 = lambda t: v_0

        desp_dtheta = self.deps_dtheta(theta, sigma, epsilon)
        dsig_dtheta = self.dsigma_dtheta(theta, sigma, epsilon)
        dv0_dtheta = self.dv_dtheta(0, theta, sigma, epsilon)

        return [desp_dtheta, dsig_dtheta, dv0_dtheta]


    def One_Strip_Solve_Sonic(self, epsilon_0, print_sol=True):

        # Start Stop Theta [0 -> 1.125] (paper)
        theta_range = [0, 1.125]

        # Initial Guesses
        sigma_0 = np.pi/2
        v0_0 = 0
        
        IGs = [epsilon_0, sigma_0, v0_0]

        # Event: Check Singular "Sonic Line"
        def v0_pre_sonic(t, y):          
            return y[2] - abs(np.sqrt((self.gamma - 1)/(self.gamma + 1)))*self.cutoff
        # Stop when reached 0.8*M=1
        v0_pre_sonic.terminal = True

        # Run Integrate Solver
        Flow_Solution = sci.integrate.solve_ivp(self.One_Strip_Sys, theta_range, IGs, events=[v0_pre_sonic], dense_output=True) 

        # Update v_0 as a function of theta for post processing
        self.v_0 = lambda t: Flow_Solution.sol(t)[2]

        # Find theta at v0 = sonic
        def v0_zero(t):
            return Flow_Solution.sol(t)[2] - abs(np.sqrt((self.gamma - 1)/(self.gamma + 1)))

        if len(Flow_Solution.t_events[0]) == 0:
            print('No theta sonic found -> lower esp by 1%...')
            return None

        try:
            theta_sonic = sci.optimize.root_scalar(v0_zero, x0=Flow_Solution.t_events[0][0], x1=Flow_Solution.t_events[0][0]*1.5, method='secant')
            if theta_sonic.converged == False or np.isclose(Flow_Solution.sol(theta_sonic.root)[2], np.sqrt((self.gamma - 1)/(self.gamma + 1))) == False or theta_sonic.root < 0 or theta_sonic.root > 1:
                print('v_0 sonic not convering -> lower esp by 1%...')
                return None
        except IndexError:
            print('v_0 sonic not found -> lower esp by 1%...')
            return None

        # E0 at theta_sonic
        E0 = self.E(0, theta_sonic.root, Flow_Solution.sol(theta_sonic.root)[1], Flow_Solution.sol(theta_sonic.root)[0])

        # Print sonic line information
        if print_sol==True:
            print(f'Eps_0: {epsilon_0:5.6f}\ttheta_s: {theta_sonic.root:6.4f} v0_conv: {theta_sonic.converged} v0_s: {self.v_0(theta_sonic.root):6.4f} E0:{E0:5.4e}')

        return Flow_Solution, E0, theta_sonic

    
    def One_Strip_Find_epsilon_0(self):

        # Start from esp_0 guess backward until negative
        esp0 = self.epsilon_0()
        E0 = 1
        d_esp = .0001
        print(f"Aescending From Initial Epsilon Guess, step:{d_esp}...")

        while E0 > 0 or np.isnan(E0):
            solver_out = self.One_Strip_Solve_Sonic(esp0, print_sol=True)

            if solver_out == None:
                esp0 *= 0.99
                        
            else:
                esp0_brak = [esp0-d_esp, esp0]
                esp0 += d_esp
                try:
                    E0 = solver_out[1]
                except TypeError:
                    continue

        print(f"Epsilon_0 Bound Founded: [{esp0_brak[0]:5.6f} {esp0_brak[1]:5.6f}]\n\nRun Bisect for Root...")

        # Now Run Bisect for root
        def E0_func(esp0):
            try:
                Flow_Solution, E0, theta_sonic = self.One_Strip_Solve_Sonic(esp0, print_sol=True)
            except TypeError:
                return 0.1

            if theta_sonic.converged == False or np.isnan(E0):
                return 0.1
            else:
                return E0

        sol = sci.optimize.root_scalar(E0_func, bracket = esp0_brak, method='bisect')

        esp_0 = sol.root
        
        print(f'Mach: {self.M_inf:3.2f} Epsilon_0: {sol.root}')
        return esp_0

    
    def One_Strip_Solve_Full(self, esp_0, theta_lis, print_sol=True):
        
        # Solution Before sonic
        sol_pre_sonic,_,theta_sonic = self.One_Strip_Solve_Sonic(esp_0, print_sol=False)

        # Solution from 0 to sonic using dense output
        theta_lis = sorted(np.append(theta_lis, theta_sonic.root))

        # Solution after sonic line theta_lis[theta_lis.index(theta_sonic.root)+1]
        theta_range_aft_sonic = [theta_sonic.root+0.001, theta_lis[-1]]
        Initial_Condition = list(sol_pre_sonic.sol(theta_sonic.root+0.001))

        sol_aft_sonic = sci.integrate.solve_ivp(self.One_Strip_Sys, theta_range_aft_sonic, Initial_Condition, dense_output=True)

        # Return Dense Output functions, pack pre and aft sonic line
        def switch_func(t, index):
            index = int(index)
            if t <= theta_sonic.root:
                return sol_pre_sonic.sol(t)[index]
            elif t > theta_sonic.root:
                return sol_aft_sonic.sol(t)[index]

        def epsilon(t):
            index=0
            return switch_func(t, index) if np.shape(t) == () else np.array(list(map(switch_func, t, index*np.ones(len(t)))))
        
        def sigma(t):
            index=1
            return switch_func(t, index) if np.shape(t) == () else np.array(list(map(switch_func, t, index*np.ones(len(t)))))


        # Define internal v_0(theta) for post process
        self.v_0 = lambda t: switch_func(t, 2) if np.shape(t) == () else np.array(list(map(switch_func, t, 2*np.ones(len(t)))))

        # Solved functions
        v1 = self.v(1, theta_lis, sigma(theta_lis))
        p1 = self.p(1, theta_lis, sigma(theta_lis), epsilon(theta_lis))
        p0 = self.p(0, theta_lis, sigma(theta_lis), epsilon(theta_lis))

        # Create dicitonary data
        data_dict = {'theta':list(theta_lis), 'epsilon':list(epsilon(theta_lis)), 'sigma':list(sigma(theta_lis)), 'kai': list(np.pi/2 - sigma(theta_lis)), 'v0':list(self.v_0(theta_lis)), 'v1':list(v1), 'p1':list(p1), 'p0':list(p0), 'p0/p0(0)': list(p0/p0[0])}

        # write to json
        self.result_file = f'data/N{self.N}_M{int(self.M_inf)}_cut{int(self.cutoff*100)}_result.json'
        with open(self.result_file, 'w') as file:
            js.dump(data_dict, file, indent=4)

        # Print Solution from functions
        if print_sol == True:
            print('\nSolving Full Solution...')
            result_lis = [theta_lis, epsilon(theta_lis), sigma(theta_lis), self.v_0(theta_lis), v1, p0, p1]

            print(f'Theta\tEpsilon\tSigma\tv0\tv1\tp0\tp1', end = '')
            for theta, epsilon_rslt, sigma_rslt, v0_rslt, v1, p0, p1 in zip(*result_lis):
                if theta == theta_sonic.root:
                    print(f'\033[0;31m \n{theta:5.4f}\t{epsilon_rslt:5.4f}\t{sigma_rslt:5.4f}\t{v0_rslt:5.4f}\t{v1:5.4f}\t{p0:5.4f}\t{p1:5.4f}\x1b[0m', end = '')
                else:
                    print(f'\n{theta:5.4f}\t{epsilon_rslt:5.4f}\t{sigma_rslt:5.4f}\t{v0_rslt:5.4f}\t{v1:5.4f}\t{p0:5.4f}\t{p1:5.4f}', end = '')

        return epsilon, sigma, self.v_0, theta_sonic.root


    def plot_properties(self, *vars):
        
        file = open(self.result_file)
        resutls = js.load(file)

        figure_lis = {}
        for i, var in enumerate(vars):
            fig, ax = plt.subplots(len([var]))
            ax.plot(resutls['theta'], resutls[var], '-o',color='black',linewidth=0.5, label=f'N={self.N} cut:{int(100*self.cutoff)}%',markersize=3)
            ax.grid(True)
            ax.set_xlabel('theta [rad]')
            ax.set_ylabel(var)
            ax.set_title(f'N{self.N} M{int(self.M_inf)} Cutoff: {int(100*self.cutoff)}%')
            ax.legend()

            figure_lis[var] = fig

        file.close()
        return figure_lis

    
    def plot_compare(self, figures:dict, Mach=3, N=1, fig_lable='--*'):

        vars = list(figures.keys())
        if Mach == 3 and N == 1:
            file = open('data/M3_epsilon_N1-3_OMB1958.json')
            OMB_shock_geo = js.load(file)
            data_dict = OMB_shock_geo['N=1']
        else:
            file = open(f'data/N3_M345_shock_geo_OMB1959.json')
            OMB_shock_geo = js.load(file)
            data_dict = OMB_shock_geo[f'Mach={Mach}']
        
        for var in vars:
            if var in data_dict:
                figures[var].axes[0].plot(data_dict['theta'], data_dict[var], fig_lable,color='black',linewidth=0.5, label=f'OMB N={N}',markersize=3)
                figures[var].axes[0].legend()
            else:    
                pass

        file.close()
        return figures


    def Two_Strip_Sys(self, theta, unks: list):

        epsilon = unks[0]
        sigma = unks[1] 
        v_0 = unks[2]
        u_2 = unks[3]
        v_2 = unks[4]
        psi_2 = unks[5]
        psi_1 = unks[6]

        kai = -sigma + np.pi/2

        # Keep format consistent
        self.v_0 = lambda t: v_0
        self.v_2 = lambda t: v_2
        self.u_2 = lambda t: u_2
        self.psi_2 = lambda t: psi_2

        # Terms for phi to avoid recrusion maximum
        gamma = self.gamma
        omega = ((self.w_inf**2)/(1 - self.w_inf**2))*(np.sin(sigma)**2)

        domega_dsigma = ((self.w_inf**2)*(2*np.sin(sigma)*np.cos(sigma)))/(1 - self.w_inf**2)

        dphi1_dsigma = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*((1 - (gamma/(omega+1)))*domega_dsigma*((1+(1/omega))**gamma) - (((gamma-1)**2)/(4*gamma))*(-gamma*(1/omega**2)*((1+(1/omega))**(gamma-1))*domega_dsigma))

        # Initialize Pressure and Density for phi2(0)
        if theta == 0:
            p2_0 = self.p(index=2, theta=0, sigma=sigma_0, epsilon=epsilon_0)
            rho2_0 = self.rho(index=2, theta=0, sigma=sigma_0, epsilon=epsilon_0)
            self.phi2_0 = p2_0/(rho2_0**gamma)

        # Get derivative for quadratic interpolation
        if theta > 0:
            if np.isnan(self.dsigma_dtheta(theta, sigma, epsilon)):
                print('')
            self.dphi1_dpsi1 = dphi1_dsigma*self.dsigma_dtheta(theta, sigma, epsilon)*(1/self.dpsi_dtheta(1, theta, sigma, epsilon))   
        else:
            self.dphi1_dpsi1 = 0

        if theta == 0.1250 or theta == 0:
            print(f'theta: {theta} kai: {kai}')
            print("    p      rho     phi      psi      s")

            print(f'0: {self.p(0, theta, sigma,epsilon):5.4f}, {self.rho(0, theta, sigma,epsilon):5.4f}, {self.phi(0, theta, sigma,epsilon):5.4f}, {self.psi(0, theta, epsilon)}, {self.s(0, theta, sigma, epsilon)}')

            print(f'2: {self.p(2, theta, sigma,epsilon):5.4f}, {self.rho(2, theta, sigma,epsilon):5.4f}, {self.phi(2, theta, sigma,epsilon):5.4f}, {self.psi(2, theta, epsilon)}, {self.s(2, theta, sigma, epsilon)}')

            print(f'1: {self.p(1, theta, sigma,epsilon):5.4f}, {self.rho(1, theta, sigma,epsilon):5.4f}, {self.phi(1, theta, sigma,epsilon):5.4f}, {self.psi(1, theta, epsilon)}, {self.s(1, theta, sigma, epsilon)}')

            print(" ")

        # Derivatives        
        du2_dtheta = self.du_dtheta(2, theta, sigma, epsilon)
        dv0_dtheta = self.dv_dtheta(0, theta, sigma, epsilon)
        dv2_dtheta = self.dv_dtheta(2, theta, sigma, epsilon)
        desp_dtheta = self.deps_dtheta(theta, sigma, epsilon)
        dsig_dtheta = self.dsigma_dtheta(theta, sigma, epsilon)
        dpsi2_dtheta = self.dpsi_dtheta(2, theta, sigma, epsilon)
        dpsi1_dtheta = self.dpsi_dtheta(1, theta, sigma, epsilon)
        
        return [desp_dtheta, dsig_dtheta, dv0_dtheta, du2_dtheta, dv2_dtheta, dpsi2_dtheta, dpsi1_dtheta]

    
    def Two_Strip_Solve(self, epsilon_0, u2_0):

        # Initial Guesses for sonic condition
        epsilon_0 = 0.708 #self.epsilon_0()
        u2_0 = -0.122 #self.u(1, 0, np.pi/2)/2

        # Initial Guess of Grouped Terms
        s1_0, s2_0, t0_0, t1_0, t2_0 = 0, 0, 0, 0, 1e-8

        # Initial conditions
        v0_0, v2_0, psi2_0 = 0, 1e-8, 0
        sigma_0 = np.pi/2

        # Theta range
        theta_range = [0, 0.1250]
        Initial_Condition = [epsilon_0, sigma_0, v0_0, u2_0, v2_0, psi2_0, 0]#, s1_0, s2_0[s1_0, , t0_0, t1_0, t2_0]

        Flow_Solution = sci.integrate.solve_ivp(self.Two_Strip_Sys, theta_range, Initial_Condition)#, first_step=0.00001, atol = 1, rtol = 1)
        
        return
    

    def Two_Strip_Interp(self, x, y, dydx):
        '''
        Format:
        x = [x0 x1]
        y = [y0 y1]
        dydx = [dy0dx0 dy1dx1]
        xtarg = x2
        '''
        # B.C 1st derivatives
        left = (1, dydx[0])
        right = (1, dydx[1])

        # Spline Interpolation from surface and wave
        # func = sci.interpolate.CubicSpline(x,y, bc_type=(left,right))

        # 1-D
        func = sci.interpolate.UnivariateSpline(x,y, k=1)

        d_func = func.derivative(1)

        # y2 = func(x_targ)

        return func, d_func



def Blunt_E0_vs_esp0_test(Mach=[]): 

    fig, axs = plt.subplots(len(Mach), figsize=(9,16))
    fig.suptitle('E0 v.s. \u03B5_0')

    for i, M in enumerate(Mach):
        case1 = Frozen_Blunt_Flow(M, 1.4, 1)
        E_0_lis = []
        esp_0 = case1.epsilon_0()

        # Plot esp0 vs E0
        if M == 3:
            esp_0_lis = np.arange(esp_0*0.97, esp_0*1.01, 0.0001)
        else:
            esp_0_lis = np.arange(esp_0*0.95, esp_0*0.98, 0.0001)
        
        E0 = 1
        for esp in esp_0_lis:
            # if E0 > 0:
            _, E0, theta_sonic = case1.One_Strip_Solve_Sonic(esp)
                # if theta_sonic.converged == True:
            E_0_lis.append(E0)
                # else:
                    # E_0_lis.append(float('nan'))
            # else:
            #     E_0_lis.append(float('nan'))

        E0_Initial_Guess = case1.One_Strip_Solve_Sonic(esp_0)[1]
        
        axs[i].scatter(esp_0, E0_Initial_Guess, color='r', marker='o', linewidths=3, label='Initial Guess \u03B5_0')
        axs[i].plot(esp_0_lis, E_0_lis, marker='o',label=f'Mach {M}')
        axs[i].plot(esp_0_lis, np.zeros(len(esp_0_lis)), 'k')
        axs[i].set_ylabel('E_0')
        axs[i].set_ylim([-0.1, 0.25])
        axs[i].grid()
        axs[i].legend()

    axs[-1].set_xlabel('\u03B5_0')
    return fig



def plot_function(theta, *funcs):

    fig, axs = plt.subplots(len(funcs))

    for i, func in enumerate(funcs):
        axs[i].plot(theta, func(theta), '-o')
        axs[i].grid(True)

    return fig


if __name__ == "__main__":
    # Two Strip Testing
    case1 = Frozen_Blunt_Flow(M_inf=3, gamma=1.4, N=2)

    epsilon_0 = 0.7086
    u2_0 = -0.122

    # Initial Guesses
    sigma_0 = np.pi/2
    v0_0 = 0

    case1.Two_Strip_Solve(epsilon_0, u2_0)
