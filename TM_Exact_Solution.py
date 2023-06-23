import numpy as np
import scipy as sci
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt

class Taylor_Macoll:
    
    def __init__(self, Mach, gamma, Beta):

        self.Mach = Mach
        self.gamma = gamma
        self.V_m2V_1 = (((gamma-1)/2)*(Mach**2)*((1 + ((gamma-1)/2)*Mach**2)**-1))**(-0.5)
        self.Beta = Beta


    # Define Taylor-Macoll ODEs
    def Taylor_Macoll_ODEs(theta: float, V: list):

        '''
        V = [v_r', v_theta'], non demensioned by V_max
        Theta: From wave to surface [rad]

        '''
        # Derivatives from ODEs
        dv_r = V[1]

        dv_theta = (V[0]*(V[1]**2) - ((gamma-1)/2)*(1 - V[0]**2 - V[1]**2)*(2*V[0] + V[1]*(1/np.tan(theta))))/(((gamma-1)/2)*(1 - V[0]**2 - V[1]**2) - (V[1]**2))

        return [dv_r, dv_theta]


    def Taylor_Macoll_ODEs_Modified(self, theta: float, V: list):

        '''
        V = [v_r', v_theta'], non demensioned by V_1
        Theta: From wave to surface [rad]

        '''
        gamma = self.gamma
        V_m2V_1 = self.V_m2V_1

        # Derivatives from ODEs
        dv_r = V[1]

        dv_theta = (V[0]*(V[1]**2) - ((gamma-1)/2)*(V_m2V_1**2 - V[0]**2 - V[1]**2)*(2*V[0] + V[1]*(1/np.tan(theta))))/(((gamma-1)/2)*(V_m2V_1**2 - V[0]**2 - V[1]**2) - (V[1]**2))

        return [dv_r, dv_theta]


    def frozen_IC(self):
        '''
        Frozen flow initial conditions.
        Flow properties imeediately behind oblique shock wave

        Mach: Freestream Mach Number
        Beta: Wave angle [deg]

        Assuming gamma = 1.4

        '''
        # Call Class Variables
        M_inf = self.Mach
        gamma = self.gamma
        V_m2V_1 = self.V_m2V_1
        Beta = np.deg2rad(self.Beta)

        # Check Mach angel vs wave angle
        if np.arcsin(1/M_inf) > Beta:
            print('Invalid Input: ')
            print('Wave angle must be greater than Mach angle')
            raise SystemExit(0)

        # Flow Deflection angle
        Delta = np.arctan((2*(1/np.tan(Beta))*((M_inf**2)*(np.sin(Beta)**2) - 1))/((M_inf**2)*(gamma + np.cos(2*Beta)) + 2))

        # Mach after shock
        M_2 = np.sqrt((((gamma+1)**2)*(M_inf**4)*(np.sin(Beta)**2) - 4*((M_inf**2)*(np.sin(Beta)**2) - 1)*(gamma*(M_inf**2)*(np.sin(Beta)**2) + 1))/((2*gamma*(M_inf**2)*(np.sin(Beta)**2) - (gamma-1))*((gamma-1)*(M_inf**2)*(np.sin(Beta)**2) + 2)))

        # Assign to calss attribute
        self.M2 = M_2

        # Non demensioned total velocity: v = V/V_max
        v_NonDem_Vm = ((2/((gamma-1)*(M_2**2))) + 1)**(-1/2)

        # Non demensioned total velocity: v = V/V_1
        v_NonDem_V1 = v_NonDem_Vm*V_m2V_1

        # Radial velocity
        v_r_0 = v_NonDem_V1*np.cos(Beta - Delta)

        # Angular velocity
        v_theta_0 = -v_NonDem_V1*np.sin(Beta - Delta)

        return [v_r_0, v_theta_0]


    def Taylor_Macoll_Solve(self):

        # Call out class variables
        Initial_Condition = self.frozen_IC()
        theta_range = np.deg2rad([self.Beta, 1])

        # Modify Inputs
        def ODEtoSolve(theta: float, V: list):
            return self.Taylor_Macoll_ODEs_Modified(theta, V)

        # Sovle ODEs
        solution = sci.integrate.solve_ivp(ODEtoSolve, theta_range, Initial_Condition, dense_output=True)

        def interp_v_theta(theta):
            """
            Transform dense output solution to v_theta as a function of theta
            """        
            return solution.sol(theta)[1]

        # Narrow root scalar search bound
        theta_low_bound = theta_range[0]
        while solution.sol(theta_low_bound)[1] <= 0:
            theta_low_bound -= 0.01

        # Interpolated function
        theta_cone = sci.optimize.root_scalar(interp_v_theta, bracket=[theta_range[0], theta_low_bound], method='bisect')
        
        # Cone surface properties non dem by V_1
        # v_r_cone = solution.sol(theta_cone.root)[0]
        # v_theta_cone = solution.sol(theta_cone.root)[1]

        # M_2_cone = np.sqrt((2/(gamma-1))*(((v_r_cone/V_m2V_1)**2)/(1-((v_r_cone/V_m2V_1)**2))))

        # theta_cone_deg = np.rad2deg(theta_cone.root)

        return [solution.sol, np.rad2deg(theta_cone.root)]


    def Taylor_Macoll_Post(self, solution_function, theta):
        '''
        Return flow properties at any input theta angle [deg]
        '''
        # Inside the shock
        M_inf = self.Mach
        M_shock = self.M2
        V_m2V_1 = self.V_m2V_1
        gamma = self.gamma
        Beta = np.deg2rad(self.Beta)

        # Get wave angle and specified angle [rad]
        theta = np.deg2rad(theta)
        Beta = solution_function.t_max

        # Flow properties as functions of theta
        v_r_theta = solution_function(theta)[0]
        v_theta_theta = solution_function(theta)[1]

        # Mach at specified theta (V_max based)
        M_theta_Vm = np.sqrt((2/(gamma-1))*(((v_r_theta/V_m2V_1)**2 + (v_theta_theta/V_m2V_1)**2)/(1-((v_r_theta/V_m2V_1)**2) - ((v_theta_theta/V_m2V_1)**2))))

        # Density behind shock (V_1 based)
        m = (M_inf**2)*(np.sin(Beta))**2
        rho_del = (6*m)/(5 + m)
        # Pressure at strip delta
        p_del = (35*m - 5)/(42*(M_inf**2))

        # Density from isentropic relations (behind shock -> cone surface, V_max based)
        rho_2_rho_1 = ((1 + ((gamma-1)/2)*(M_shock**2))/(1 + ((gamma-1)/2)*(M_theta_Vm**2)))**(-1/(1-gamma))
        rho_theta = rho_del*rho_2_rho_1

        # Pressure Ratio
        p_2_p_1 = rho_2_rho_1**gamma
        p_theta = p_del * p_2_p_1

        # temperature
        T_theta = gamma*(M_inf**2)*p_theta/rho_theta

        return [v_r_theta, v_theta_theta, rho_theta, p_theta, T_theta]


    def coord_trans(V_r, V_theta, angle):
        '''
        angle [rad]

        ***Keep for code archive***
        
        '''
        angle = np.deg2rad(angle)

        V_x = V_r*np.cos(angle) - V_theta*np.sin(angle)
        V_y = V_r*np.sin(angle) + V_theta*np.cos(angle)

        return [V_x, V_y]


def Taylor_Maccoll_Solve_All(Mach, gamma, Beta):
    '''
    ***Keep for code archive***
    '''
     
    case = Taylor_Macoll(Mach, gamma, Beta)
    # Stop angle for solver
    stop_theta = 1
    
    theta_range = [Beta, stop_theta]
    Initial_Condition = Taylor_Macoll.frozen_IC(case)

    # Get Full Solution with theta range
    [sol_func, theta_cone] = Taylor_Macoll.Taylor_Macoll_Solve(case)

    # Define intermediate strip angle [deg]
    theta_1 = (Beta+theta_cone)/2

    # Get flow properties at half theta line for 2-strip
    # [v_r_1, v_theta_1, rho_1] = Taylor_Macoll.Taylor_Macoll_Post(sol_func, theta_1)

    # Transfer from polar coord to cone x-y coord
    # [u_1, v_1] = Taylor_Macoll.coord_trans(v_r_1, v_theta_1, theta_1 - theta_cone)

    # Get flow properties at cone
    [u_0, v_0, rho_0, p_0, T_0] = Taylor_Macoll.Taylor_Macoll_Post(case, sol_func, theta_cone)


    # Print Information
    print(f'\nInput Wave Angle: {Beta:5.4f} [deg]')
    print(f'Taylor Macoll Solution: {theta_cone:5.4f} [deg]')
    # print(f'Flow Properties at 1/2 theta: {theta_1} in Polar Coord')
    # print(f'v_r_1: {v_r_1} \nv_theta_1: {v_theta_1} \nrho_1: {rho_1}\n')

    # print(f'Flow Properties at 1/2 theta: {theta_1} in x-y Coord')
    # print(f'u_1: {u_1} \nv_1: {v_1} \nrho_1: {rho_1}\n')
    print(f'Flow Properties at cone in x-y Coord')
    print(f'u_0: {u_0:5.4f} \nv_0: {v_0:5.4f} \nrho_0: {rho_0:5.4f}\np_0: {p_0:5.4f}\nT_0: {T_0:5.4f}\nS')

    return

