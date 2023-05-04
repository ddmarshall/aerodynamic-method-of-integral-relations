from Method_of_Integral_Relations import Frozen_Blunt_Flow as FBF
import Method_of_Integral_Relations as MIR
import numpy as np
from matplotlib import pyplot as plt
import tikzplotlib as mat2tik

MIR.Plot_Compare_Cutoff(Mach=3)

# MIR.One_Strip_Single_Case(Mach=3, cutoff=0.95, plot_compare=True, print_sol=False)

# MIR.One_Strip_Single_Case(Mach=4, cutoff=0.95)

# MIR.One_Strip_Single_Case(Mach=5, cutoff=0.95)
# Mach = 3

# case = FBF(M_inf=Mach, gamma=1.4, N=1, cutoff=0.95)

# # Solve stand off distance
# solved_esp_0 = case.One_Strip_Find_epsilon_0()

# # Run full solution with given theta array for out put
# theta_lis = np.arange(0, 1.250, 0.0625)

# # Solved unkown functions as dense output
# [epsilon, sigma, v0, theta_v0s, theta_w1s, theta_lis]=case.One_Strip_Solve_Full(solved_esp_0, theta_lis)

# case.plot_contour(epsilon, sigma, v0, theta_lis[theta_lis.index(theta_v0s)], theta_lis[theta_lis.index(theta_w1s)])

# resutls_figures = case.plot_properties('epsilon','v0','p0/p0(0)','kai')

# # Add comparation to existing plots
# case.plot_compare(resutls_figures, Mach=Mach, N=3)
# if Mach == 3:
#     case.plot_compare(resutls_figures, Mach=Mach, N=1, fig_lable='--^')

plt.show()
# mat2tik.save("test.tikz", axis_height='9cm', axis_width='12cm')


