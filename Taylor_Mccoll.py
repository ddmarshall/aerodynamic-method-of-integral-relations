import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate


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


def Taylor_Macoll_ODEs_Modified(theta: float, V: list):

    '''
    V = [v_r', v_theta'], non demensioned by V_1
    Theta: From wave to surface [rad]

    '''
    # V_max/V_1
    v_m2V_1 = (((gamma-1)/2)*(Mach**2)*((1 + ((gamma-1)/2)*Mach**2)**-1))**(-0.5)

    # Derivatives from ODEs
    dv_r = V[1]

    dv_theta = (V[0]*(V[1]**2) - ((gamma-1)/2)*(v_m2V_1**2 - V[0]**2 - V[1]**2)*(2*V[0] + V[1]*(1/np.tan(theta))))/(((gamma-1)/2)*(v_m2V_1**2 - V[0]**2 - V[1]**2) - (V[1]**2))

    return [dv_r, dv_theta]


def frozen_IC(M_inf: float, Beta: float):
    '''
    Frozen flow initial conditions.
    Flow properties imeediately behind oblique shock wave

    Mach: Freestream Mach Number
    Beta: Wave angle [deg]

    Assuming gamma = 1.4

    '''
    global M_2
    
    Beta = np.deg2rad(Beta)

    # Check Mach angel vs wave angle
    if np.arcsin(1/M_inf) > Beta:
        print('Invalid Solution: ')
        print('Wave angle must be greater than Mach angle')

    # Flow Deflection angle
    Delta = np.arctan((2*(1/np.tan(Beta))*((M_inf**2)*(np.sin(Beta)**2) - 1))/((M_inf**2)*(gamma + np.cos(2*Beta)) + 2))

    # Mach after shock
    M_2 = np.sqrt((((gamma+1)**2)*(M_inf**4)*(np.sin(Beta)**2) - 4*((M_inf**2)*(np.sin(Beta)**2) - 1)*(gamma*(M_inf**2)*(np.sin(Beta)**2) + 1))/((2*gamma*(M_inf**2)*(np.sin(Beta)**2) - (gamma-1))*((gamma-1)*(M_inf**2)*(np.sin(Beta)**2) + 2)))

    # Non demensioned total velocity: v = V/V_max
    v_NonDem_Vm = ((2/((gamma-1)*(M_2**2))) + 1)**(-1/2)

    # V_max/V_1
    v_m2V_1 = (((gamma-1)/2)*(M_inf**2)*((1 + ((gamma-1)/2)*M_inf**2)**-1))**(-0.5)

    # Non demensioned total velocity: v = V/V_1
    v_NonDem_V1 = v_NonDem_Vm*v_m2V_1

    # Radial velocity
    v_r_0 = v_NonDem_V1*np.cos(Beta - Delta)

    # Angular velocity
    v_theta_0 = -v_NonDem_V1*np.sin(Beta - Delta)

    return [v_r_0, v_theta_0]


def Taylor_Macoll_Solve(ODEs, Initial_Condition, theta_range):

    theta_range = np.deg2rad(theta_range)

    solution = sci.integrate.solve_ivp(ODEs, theta_range, Initial_Condition, dense_output=True)

    theta_range_list = np.linspace(theta_range[0], theta_range[1], num=100)

    def interp_v_theta(theta):
        """
        Transform dense output solution to v_theta as a function of theta
        """        
        return solution.sol(theta)[1]

    # Interpolated function
    theta_cone = sci.optimize.root_scalar(interp_v_theta, bracket=[theta_range[0], theta_range[1]], method='bisect')
    
    # Cone surface properties
    v_r_cone = solution.sol(theta_cone.root)[0]
    v_theta_cone = solution.sol(theta_cone.root)[1]
    M_2_cone = np.sqrt((2/(gamma-1))*((v_r_cone**2)/(1-(v_r_cone**2))))

    theta_cone_deg = np.rad2deg(theta_cone.root)

    return [solution.sol, np.rad2deg(theta_cone.root)]


def Taylor_Macoll_Post(solution_function, theta):
    '''
    Return flow properties at any input theta angle [deg]
    '''
    # Inside the shock
    M_inf = Mach
    M_shock = M_2

    theta = np.deg2rad(theta)
    Beta = solution_function.t_max
    
    v_r_shock = solution_function(Beta)[0]
    # Cone surface properties as a fundtion of theta
    v_r_theta = solution_function(theta)[0]
    M_theta = np.sqrt((2/(gamma-1))*((v_r_theta**2)/(1-(v_r_theta**2))))

    # Density behind shock
    m = (M_inf**2)*(np.sin(Beta))**2
    rho_del = (6*m)/(5 + m)

    # Density from isentropic relations
    rho_2_rho_1 = ((1 + ((gamma-1)/2)*(M_theta**2))/(1 + ((gamma-1)/2)*(M_inf**2)))**(-1/(1-gamma))
    
    #(v_r_theta/v_r_shock)**(gamma-1)
    rho_theta = rho_del*rho_2_rho_1

    return


def main():

    global Mach
    global gamma

    Mach = 8
    gamma = 1.4
    Beta = 38.155
    stop_theta = 30
    
    theta_range = [Beta, stop_theta]
    Initial_Condition = frozen_IC(Mach, Beta)

    # Get Full Solution with theta range
    [sol_func, theta_cone] = Taylor_Macoll_Solve(Taylor_Macoll_ODEs_Modified, Initial_Condition, theta_range)

    # Get flow properties at specified theta line
    Taylor_Macoll_Post(sol_func, theta_cone)
    
    print(theta_cone)

    '''
    # Plots
    # plt.plot(theta, v_theta)
    # plt.plot(theta, v_r)
    # plt.title('Mach: ' + str(Mach) + ' Wave Angle: ' + str(Beta))
    # plt.xlabel('Theta [deg]')
    # plt.legend(['v_theta', 'v_r'])
    # plt.grid()
    # plt.show()
    '''

    return


main()


