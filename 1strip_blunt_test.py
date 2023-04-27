import Method_of_Integral_Relations as MIR
from Method_of_Integral_Relations import Frozen_Blunt_Flow as FBF
import numpy as np
import json as js



case = FBF(M_inf=3, gamma=1.4, N=1, cutoff=0.93)

# Solve stand off distance
solved_esp_0 = case.One_Strip_Find_epsilon_0()

# Run full solution with given theta array for out put
theta_lis = np.arange(0, 1.125, 0.0625)

# Solved unkown functions as dense output
case.One_Strip_Solve_Full(solved_esp_0, theta_lis)

resutls_figures = case.plot_properties('epsilon','v0','p0/p0(0)','kai')

case.plot_compare(resutls_figures, Mach=3, N=3)
case.plot_compare(resutls_figures, Mach=3, N=1, fig_lable='--^')

pass
# # Create dicitonary data
# shock_geo = {'theta':list(theta_lis), 'epsilon':list(epsilon(theta_lis)), 'sigma':list(sigma(theta_lis)), 'v0':list(v0(theta_lis))}

# # Pressure data
# shock_geo["p0/p0(0)"] = []
# for t in theta_lis:
#     shock_geo["p0/p0(0)"].append(case.p(0, t, epsilon(t), sigma(t))/case.p(0, 0, epsilon(0), sigma(0)))

# # write to json
# with open(f'data/N{case.N}_M{int(case.M_inf)}_shock_geo_cut{int(case.cutoff*100)}_result.json', 'w') as fp:
#     js.dump(shock_geo, fp, indent=4)
