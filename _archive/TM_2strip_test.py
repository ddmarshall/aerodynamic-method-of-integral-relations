from Exact_Solution import Taylor_Macoll as TM
from Method_of_Integral_Relations import Frozen_Cone_Flow as FCF
import numpy as np
import matplotlib.pyplot as plt


M_inf = 8
Beta = 38.155
gamma = 1.4

# Exact Solution
Exact = TM(M_inf, gamma, Beta)
sol_func, Cone_Angle_Exact = TM.Taylor_Macoll_Solve(Exact)

# Define intermediate strip angle [deg]
theta_1 = (Beta + Cone_Angle_Exact)/2

# Get flow properties at half theta line for 2-strip
[v_r_1, v_theta_1, rho_1, p_1] = TM.Taylor_Macoll_Post(Exact, sol_func, theta_1)

# Transfer from polar coord to cone x-y coord
[u_1, v_1] = TM.coord_trans(v_r_1, v_theta_1, theta_1 - Cone_Angle_Exact)

# Get flow properties at cone theta for 2-strip
[u_0, v_0, rho_0, p_0] = TM.Taylor_Macoll_Post(Exact, sol_func, Cone_Angle_Exact)

# One-Strip Solution
MIR_1 = FCF(M_inf, gamma, Beta)
Cone_Angle_1st = FCF.one_strip_solve(MIR_1, full_output=True)

# Print Information
print(f'\nInput Wave Angle: {Beta} [deg]')
print(f'Taylor Macoll Solution: {Cone_Angle_Exact} [deg]\n')
print(f'Flow Properties at 1/2 theta: {theta_1} in Polar Coord')
print(f'v_r_1: {v_r_1} \nv_theta_1: {v_theta_1} \nrho_1: {rho_1} \np_1: {p_1}\n')

print(f'Flow Properties at 1/2 theta: {theta_1} in x-y Coord')
print(f'u_1: {u_1} \nv_1: {v_1} \nrho_1: {rho_1} \np_1: {p_1}\n')

print(f'Flow Properties at cone: {Cone_Angle_Exact} in x-y Coord')
print(f'u_0  : {u_0} \nv_0  : {v_0} \nrho_0: {rho_0} \np_0: {p_0}\n')

print('1-strip full solutions:\n')
print('Cone Angle, u_del, u_0, v_del, p_0, p_del, T_0, T_del, rho_0, rho_del')
print(Cone_Angle_1st)

# Two-Strip Solution
Cone_Angle_2nd, solver_msg = FCF.two_strip_solve(MIR_1, full_output=False)