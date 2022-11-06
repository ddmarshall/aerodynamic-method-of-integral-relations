import numpy as np
import scipy as sci
from scipy import optimize


def frozen_one_strip_cone(M_inf: float, Beta: float, gamma: float):
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
    Beta = np.deg2rad(Beta)

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

    # Perform solving
    theta = sci.optimize.fsolve(solve_theta, [0.1])[0]

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
  
    return np.rad2deg(theta), u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0

# Testing
Mach = 4
Wave_angle = 37.8
gamma = 1.4

[theta, u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0] = frozen_one_strip_cone(Mach, Wave_angle, gamma)

print('1-Strip Conical Frozen Flow\n')
print('Condition:\nM_inf: ' + str(Mach) + '\n' + 'Wave Angle: ' + str(Wave_angle) + ' [deg]\n' + 'Gamma: ' + str(gamma))
print('\nSolved:\nHalf Cone Angle: ' + str(theta) + ' [deg]')