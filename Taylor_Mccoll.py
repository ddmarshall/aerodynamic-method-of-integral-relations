import numpy as np
import scipy as sci
from scipy import optimize
from scipy import integrate


# Define Taylor-Mccoll ODEs
def Taylor_Mccoll_ODEs(theta: float, V: list):

    '''
    V = [v_r', v_theta'], non demensioned
    Theta: From wave to surface

    '''
    theta = np.deg2rad(theta)

    # Gas constant ratio
    gamma = 1.4

    # Derivatives from ODEs
    dv_r = V[1]

    dv_theta =  (-(2*V[0] + V[1]*(1/np.tan(theta)))*((gamma-1)/2)*(1 - V[0]**2 - V[1]**2) + V[0]*(V[1]**2))/(((gamma-1)/2)*(1 - V[0]**2 - V[1]**2) - (V[1]**2))

    return [dv_r, dv_theta]


def frozen_IC(M_inf: float, Beta: float):
    '''
    Frozen flow initial conditions.
    Flow properties imeediately behind oblique shock wave

    Mach: Freestream Mach Number
    Beta: Wave angle [deg]

    Assuming gamma = 1.4

    '''
    Beta = np.deg2rad(Beta)

    v_r_0 = np.cos(Beta)
    
    v_theta_0 = -(((M_inf**2)*(np.sin(Beta)**2) + 5)/(6*(M_inf**2)*np.sin(Beta)))

    return [v_r_0, v_theta_0]


def Taylor_Mccoll_Solve(ODEs, Initial_Condition, Wave_Angle):

    theta_range = [np.deg2rad(Wave_Angle), np.deg2rad(20)]

    solution = sci.integrate.solve_ivp(ODEs, theta_range, Initial_Condition)

    return solution


def main():

    Mach = 3
    Beta = 40

    Initial_Condition = frozen_IC(Mach, Beta)

    Taylor_Mccoll_Solve(Taylor_Mccoll_ODEs, Initial_Condition, Beta)

    return


main()


