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

    return theta, u_del, v_del, p_del, T_del, rho_del, u_0, p_0, T_0, rho_0


# Solving 8 eqns 8 unkws
def two_strip_relations(unkns):
    
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
    m = (M_inf**2)*(np.sin(Beta))**2
    a = 5*(m-1)/(6*M_inf**2)

    # Density at strip_delta
    rho_del = (6*m)/(5 + m)

    # Pressure at strip delta
    p_del = (35*m - 5)/(42*(M_inf**2))

    # Temperature at strip delta using state equation
    T_del = gamma*(M_inf**2)*p_del/rho_del

    #  u_del, v_del from frozen shock relations
    u_del = (1-a)*np.cos(theta) + a*np.sin(theta)*cot_beta
    v_del = -(1-a)*np.sin(theta) + a*np.cos(theta)*cot_beta

    # Energy and state equations
    p_0 = (rho_0/(gamma*(M_inf**2)))*(1 + (1 - u_0**2)*((gamma-1)/2)*(M_inf**2))
    p_1 = (rho_1/(gamma*(M_inf**2)))*(1 + (1 - (u_1**2) - (v_1**2))*((gamma-1)/2)*(M_inf**2))

    # Integral Relation Unknows
    # Q Terms
    Q_0 = [(rho_0*u_0)*r_0, (p_0 + rho_0*(u_0**2))*r_0 , 0]
    Q_1 = [(rho_1*u_1)*r_1, (p_1 + rho_1*(u_1**2))*r_1, (rho_1*u_1*v_1)*r_1]
    Q_del = [(rho_del*u_del)*r_del, (p_del + rho_del*(u_del**2))*r_del, (rho_del*u_del*v_del)*r_del]
    # G Terms
    G_0 = [0, 0, p_0*r_0]
    G_1 = [(rho_1*v_1)*r_1, (rho_1*u_1*v_1)*r_1, (p_1 + rho_1*(v_1**2))*r_1]
    G_del = [(rho_del*v_del)*r_del, (rho_del*u_del*v_del)*r_del, (p_del + rho_del*(v_del**2))*r_del]
    # F Terms
    F_0 = [0, p_0, p_0*cot_theta]
    F_1 = [0, p_1, p_1*cot_theta]
    F_del = [0, p_del, p_del*cot_theta]

    # Integrate from 0 -> del
    I0_C = (Q_0[0] - 4*Q_1[0] + 3*Q_del[0])*np.tan(Lambda) - 4*(G_0[0] - 2*G_1[0] + G_del[0]) - np.tan(Lambda)*(F_0[0] - F_del[0])
    I0_XM = (Q_0[1] - 4*Q_1[1] + 3*Q_del[1])*np.tan(Lambda) - 4*(G_0[1] - 2*G_1[1] + G_del[1]) - np.tan(Lambda)*(F_0[1] - F_del[1])
    I0_YM = (Q_0[2] - 4*Q_1[2] + 3*Q_del[2])*np.tan(Lambda) - 4*(G_0[2] - 2*G_1[2] + G_del[2]) - np.tan(Lambda)*(F_0[2] - F_del[2])

    # Integrate from 0 -> 1/2 del
    I1_C = 4*(Q_1[0] - Q_del[0])*np.tan(Lambda) - G_0[0] - 4*G_1[0] + 5*G_del[0] - np.tan(Lambda)*(2*F_1[0] + F_del[0])
    I1_XM = 4*(Q_1[1] - Q_del[1])*np.tan(Lambda) - G_0[1] - 4*G_1[1] + 5*G_del[1] - np.tan(Lambda)*(2*F_1[1] + F_del[1])
    I1_YM = 4*(Q_1[2] - Q_del[2])*np.tan(Lambda) - G_0[2] - 4*G_1[2] + 5*G_del[2] - np.tan(Lambda)*(2*F_1[2] + F_del[2])

    eqns_to_solve = np.array([I0_C, I0_XM, I0_YM, I1_C, I1_XM, I1_YM])

    print(unkns, eqns_to_solve)

    return eqns_to_solve


def frozen_two_strip_cone_solve(initial_guess):

    solutions = sci.optimize.fsolve(two_strip_relations, initial_guess, full_output=True)

    print('theta N=2: ' + str(np.rad2deg(solutions[0][-1])))
    
    return solutions[0]


def Two_Strip_Eqn_Test(input):

    # Extract Unknows
    u_0 = input[0]
    u_1 = (input[0] + input[1])/2
    u_del = input[1]

    v_1 = input[2]/2
    v_del = input[2]

    p_0 = input[3]
    p_1 = (input[3] + input[4])/2
    p_del = input[4]

    rho_0 = input[5]
    rho_1 = (input[5] + input[6])/2
    rho_del = input[6]

    theta = np.deg2rad(input[7])
    Beta = np.deg2rad(input[8])

    # Because Numpy has no cot :(
    cot_theta = 1/np.tan(theta)
    cot_beta = 1/np.tan(Beta)
    Lambda = Beta - theta
    cot_Lambda = 1/np.tan(Lambda)  

    # Define r_i
    r_0 = 1 
    r_1 = 1 + 0.5*np.tan(Lambda)*cot_theta
    r_del = 1 + np.tan(Lambda)*cot_theta

    # Energy and state equations
    # M_inf = 8
    # p_0 = (rho_0/(gamma*(M_inf**2)))*(1 + (1 - u_0**2)*((gamma-1)/2)*(M_inf**2))
    # p_1 = (rho_1/(gamma*(M_inf**2)))*(1 + (1 - (u_1**2) - (v_1**2))*((gamma-1)/2)*(M_inf**2))

    # Integral Relation Unknows
    Q_0 = [rho_0*u_0*r_0, (p_0 + rho_0*(u_0**2))*r_0 , 0]
    Q_1 = [rho_1*u_1*r_1, (p_1 + rho_1*(u_1**2))*r_1, (rho_1*u_1*v_1)*r_1]
    Q_del = [rho_del*u_del*r_del, (p_del + rho_del*(u_del**2))*r_del, (rho_del*u_del*v_del)*r_del]

    G_0 = [0, 0, 0]
    G_1 = [(rho_1*v_1)*r_1, (rho_1*u_1*v_1)*r_1, (p_1 + rho_1*(v_1**2))*r_1]
    G_del = [(rho_del*v_del)*r_del, (rho_del*u_del*v_del)*r_del, (p_del + rho_del*(v_del**2))*r_del]

    F_0 = [0, p_0, p_0*cot_theta]
    F_1 = [0, p_1, p_1*cot_theta]
    F_del = [0, p_del, p_del*cot_theta]

    # Integrate from 0 -> del
    I0_C = (Q_0[0] - 4*Q_1[0] + 3*Q_del[0])*np.tan(Lambda) - 4*(-2*G_1[0] + G_del[0]) - np.tan(Lambda)*(F_0[0] - F_del[0])
    I0_XM = (Q_0[1] - 4*Q_1[1] + 3*Q_del[1])*np.tan(Lambda) - 4*(-2*G_1[1] + G_del[1]) - np.tan(Lambda)*(F_0[1] - F_del[1])
    I0_YM = (Q_0[2] - 4*Q_1[2] + 3*Q_del[2])*np.tan(Lambda) - 4*(-2*G_1[2] + G_del[2]) - np.tan(Lambda)*(F_0[2] - F_del[2])

    # Integrate from 0 -> 1/2 del
    I1_C = 4*(Q_1[0] - Q_del[0])*np.tan(Lambda) - 4*G_1[0] + 5*G_del[0] - np.tan(Lambda)*(2*F_1[0] + F_del[0])
    I1_XM = 4*(Q_1[1] - Q_del[1])*np.tan(Lambda) - 4*G_1[1] + 5*G_del[1] - np.tan(Lambda)*(2*F_1[1] + F_del[1])
    I1_YM = 4*(Q_1[2] - Q_del[2])*np.tan(Lambda) - 4*G_1[2] + 5*G_del[2] - np.tan(Lambda)*(2*F_1[2] + F_del[2])

    return


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
    # Inital Guess
    u_0 = 0.7915200156896637 #0.7913637553006431 
    u_1 = 0.7919718861644587
    v_1 = -0.030536066611886743
    rho_0 = 5.192282199293422
    rho_1 = 5.138006805966865
    theta = 0.5847021888888834

    initial_guess =  np.array([u_0, u_1, v_1, rho_0, rho_1, theta])

    solution = frozen_two_strip_cone_solve(initial_guess)
    print(solution)

    return

main()
