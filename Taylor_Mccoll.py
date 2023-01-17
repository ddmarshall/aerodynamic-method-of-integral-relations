import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate


# Define Taylor-Mccoll ODEs
def Taylor_Mccoll_ODEs(theta: float, V: list):

    '''
    V = [v_r', v_theta'], non demensioned
    Theta: From wave to surface [rad]

    '''
    # Gas constant ratio
    gamma = 1.4

    # Derivatives from ODEs
    dv_r = V[1]

    dv_theta = (V[0]*(V[1]**2) - ((gamma-1)/2)*(1 - V[0]**2 - V[1]**2)*(2*V[0] + V[1]*(1/np.tan(theta))))/(((gamma-1)/2)*(1 - V[0]**2 - V[1]**2) - (V[1]**2))

    return [dv_r, dv_theta]


def frozen_IC(M_inf: float, Beta: float):
    '''
    Frozen flow initial conditions.
    Flow properties imeediately behind oblique shock wave

    Mach: Freestream Mach Number
    Beta: Wave angle [deg]

    Assuming gamma = 1.4

    '''
    gamma = 1.4
    Beta = np.deg2rad(Beta)

    # Check Mach angel vs wave angle
    if np.arcsin(1/M_inf) > Beta:
        print('Invalid Solution: ')
        print('Wave angle must be greater than Mach angle')

    M_n_inf = M_inf*np.sin(Beta)

    M_n_2 = np.sqrt((1 + ((gamma-1)/2)*(M_n_inf**2))/(gamma*(M_n_inf**2) - ((gamma-1)/2)))

    # Flow Deflection angle
    Delta = np.arctan((2*(1/np.tan(Beta))*((M_inf**2)*(np.sin(Beta)**2) - 1))/((M_inf**2)*(gamma + np.cos(2*Beta)) + 2))

    # Mach after shock
    M_2 = M_n_2/np.sin(Beta - Delta)

    # Non demensioned total velocity
    v = np.sqrt((((gamma-1)/2)*(M_2**2))/(1 + ((gamma-1)/2)*(M_2**2)))

    # Radial velocity
    v_r_0 = v*np.cos(Beta - Delta)
    # v_r_0 = np.cos(Beta)

    # Angular velocity
    v_theta_0 = -v*np.sin(Beta - Delta)
    # v_theta_0 = -((M_inf**2)*(np.sin(Beta)**2) + 5)/(6*(M_inf**2)*np.sin(Beta))

    v_max = (v_r_0**2 + v_theta_0**2)**0.5

    return [v_r_0, v_theta_0]


def Taylor_Mccoll_Solve(ODEs, Initial_Condition, theta_range):

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
    M_2_cone = np.sqrt((2/(1.4-1))*((v_r_cone**2)/(1-(v_r_cone**2))))

    v_r_0, v_theta_0 = Initial_Condition[0], Initial_Condition[1]

    theta_cone_deg = np.rad2deg(theta_cone.root)

    return [np.rad2deg(solution.t), solution.y[0], solution.y[1]]


def main():

    Mach = 8
    Beta = 38.155
    stop_theta = 30
    
    theta_range = [Beta, stop_theta]
    Initial_Condition = frozen_IC(Mach, Beta)

    [theta, v_r, v_theta] = Taylor_Mccoll_Solve(Taylor_Mccoll_ODEs, Initial_Condition, theta_range)

    # Plots
    plt.plot(theta, v_theta)
    plt.plot(theta, v_r)
    plt.title('Mach: ' + str(Mach) + ' Wave Angle: ' + str(Beta))
    plt.xlabel('Theta [deg]')
    plt.legend(['v_theta', 'v_r'])
    plt.grid()
    plt.show()

    return


main()


