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

    # Perform solving by guessing theta~0.1
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
    
    v_max = (u_del**2 + v_del**2)**0.5

    theta_deg = np.rad2deg(theta)

    return np.rad2deg(theta), u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0


def frozen_two_strip_cone(M_inf: float, Beta: float, gamma: float):

    
    Beta = np.deg2rad(Beta)

    # Frozen Shock Relations
    m = (M_inf**2)*(np.sin(Beta))**2
    a = 5*(m-1)/(6*M_inf**2)

    # Density at strip_delta
    rho_del = (6*m)/(5 + m)

    # Pressure at strip delta
    p_del = (35*m - 5)/(42*(M_inf**2))

    # Temperature at strip delta using state equation
    T_del = gamma*(M_inf**2)*p_del/rho_del

    # Solving 8 eqns 8 unkws
    def two_strip_relations(unkns):
        
        # Extract Unknows
        u_0 = unkns[0]
        u_1 = unkns[1]
        v_1 = unkns[2]
        #p_0 = unkns[3]
        #p_1 = unkns[4]
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

        #  u_del, v_del from frozen shock relations
        u_del = (1-a)*np.cos(theta) + a*np.sin(theta)*cot_beta
        v_del = -(1-a)*np.sin(theta) + a*np.cos(theta)*cot_beta

        # Energy and state equations
        p_0 = (rho_0/(gamma*(M_inf**2)))*(1 + (1 - u_0**2)*((gamma-1)/2)*(M_inf**2))
        p_1 = (rho_1/(gamma*(M_inf**2)))*(1 + (1 - u_1**2 - v_1**2)*((gamma-1)/2)*(M_inf**2))

        # Integral Relations

        # Integrate from 0 -> del
        #I0_C = (rho_0*u_0*r_0 + 4*(rho_1*u_1*r_1) - 5*(rho_del*u_del*r_del))*np.tan(Lambda) + 6*(rho_del*v_del)*r_del
        I0_C = (rho_0*u_0*r_0 - 4*(rho_1*u_1*r_1) + 3*(rho_del*u_del*r_del))*np.tan(Lambda) - 4*(-2*(rho_1*v_1)*r_1 + (rho_del*v_del)*r_del)

        #I0_XM = ((p_0 + rho_0*(u_0**2))*r_0 + 4*(p_1 + rho_1*(u_1**2))*r_1 - 5*(p_del + rho_del*(u_del**2))*r_del)*np.tan(Lambda) + \
        #     6*(rho_del*u_del*v_del)*r_del - np.tan(Lambda)*(p_0 + 4*p_1 + p_del)
        I0_XM = ((p_0 + rho_0*(u_0**2))*r_0 - 4*(p_1 + rho_1*(u_1**2))*r_1 + 3*(p_del + rho_del*(u_del**2))*r_del)*np.tan(Lambda) - \
             4*(-2*(rho_1*u_1*v_1)*r_1 + (rho_del*u_del*v_del)*r_del) - np.tan(Lambda)*(p_0 - p_del)

        #I0_YM = (4*(rho_1*u_1*v_1)*r_1 - 5*(rho_del*u_del*v_del)*r_del)*np.tan(Lambda) + \
        #     6*(p_del + rho_del*(v_del**2))*r_del - np.tan(Lambda)*cot_theta*(p_0 + 4*p_1 + p_del)             
        I0_YM = (-4*(rho_1*u_1*v_1)*r_1 + 3*(rho_del*u_del*v_del)*r_del)*np.tan(Lambda) - \
             4*(-2*(p_1 + rho_1*(v_1**2))*r_1 + (p_del + rho_del*(v_del**2))*r_del) - np.tan(Lambda)*cot_theta*(p_0 - p_del) 

        # Integrate from 0 -> 1/2 del
        #I1_C = (5*rho_0*u_0*r_0 - 16*(rho_1*u_1*r_1) - (rho_del*u_del*r_del))*np.tan(Lambda) + 21*(rho_1*v_1)*r_1
        I1_C = (4*(rho_1*u_1*r_1) - 4*(rho_del*u_del*r_del))*np.tan(Lambda) - 4*(rho_1*v_1)*r_1 + 5*(rho_del*v_del)*r_del

        #I1_XM = (5*(p_0 + rho_0*(u_0**2))*r_0 - 16*(p_1 + rho_1*(u_1**2))*r_1 - (p_del + rho_del*(u_del**2))*r_del)*np.tan(Lambda) + \
        #     21*(rho_1*u_1*v_1)*r_1 - np.tan(Lambda)*(5*p_0 + 8*p_1 - p_del)
        I1_XM = (4*(p_1 + rho_1*(u_1**2))*r_1 - 4*(p_del + rho_del*(u_del**2))*r_del)*np.tan(Lambda) - \
             4*(rho_1*u_1*v_1)*r_1 + 5*(rho_del*u_del*v_del)*r_del - np.tan(Lambda)*(2*p_1 + p_del)

        #I1_YM = (-16*(rho_1*u_1*v_1)*r_1 - (rho_del*u_del*v_del)*r_del)*np.tan(Lambda) + \
        #     21*(p_1 + rho_1*(v_1**2))*r_1 - np.tan(Lambda)*cot_theta*(5*p_0 + 8*p_1 - p_del)
        I1_YM = (4*(rho_1*u_1*v_1)*r_1 - 4*(rho_del*u_del*v_del)*r_del)*np.tan(Lambda) - \
             4*(p_1 + rho_1*(v_1**2))*r_1 + 5*(p_del + rho_del*(v_del**2))*r_del - np.tan(Lambda)*cot_theta*(2*p_1 + p_del)

        eqns_to_solve = np.array([I0_C, I0_XM, I0_YM, I1_C, I1_XM, I1_YM])

        return eqns_to_solve
    
    # Inital Guess: [u_0, u_1, v_1, rho_0, rho_1, theta]
    sol_guess =  np.array([0.7, 0.81, -0.04, 6, 5, 0.5])

    solutions = sci.optimize.fsolve(two_strip_relations, sol_guess, full_output=True, xtol=1.49012e-10, maxfev=500)

    print('theta N=2: ' + str(np.rad2deg(solutions[0][-1])))

    return



#  1-Strip - Testing
Mach = 8
Wave_angle = 38.155
gamma = 1.4

[theta, u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0] = frozen_one_strip_cone(Mach, Wave_angle, gamma)
print('theta N=1: ' + str(theta))
print('u_0: ' + str(u_0))
print('rho_0: ' + str(rho_0))

frozen_two_strip_cone(Mach, Wave_angle, gamma)



print('1-Strip Conical Frozen Flow\n')
#print('Condition:\nM_inf: ' + str(Mach) + '\n' + 'Wave Angle: ' + str(Wave_angle) + ' [deg]\n' + 'Gamma: ' + str(gamma))
#print('\nSolved:\nHalf Cone Angle: ' + str(theta) + ' [deg]')