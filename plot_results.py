from Exact_Solution import Taylor_Macoll as TM
from Method_of_Integral_Relations import Frozen_Cone_Flow as FCF
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4

Mach_Range = [1.5, 2, 2.5, 4, 10]

plt.figure(1)

MIR_1 = FCF(12, gamma, 40)
one_strip_sol = FCF.one_strip_solve(MIR_1, full_output=True)
Cone_Angle_1st = one_strip_sol[0]

# These can be grab from 1 strip
# u_del =  one_strip_sol[1]
# u_0 =  one_strip_sol[2]
# u_1 =  (u_0 + u_del)/2
# v_del =  one_strip_sol[3]
# v_1 = v_del/2
# rho_0 = one_strip_sol[8]
# rho_del = one_strip_sol[9]
# rho_1 = (rho_0 + rho_del)/2
# theta = np.deg2rad(one_strip_sol[0]) #0.32535828738426736 #

# initial_guess =  np.array([u_0, u_1, v_1, rho_0, rho_1, theta])
Cone_Angle_2nd = FCF.two_strip_solve(MIR_1, full_output=False)

my_labels = {"x1" : "Taylor-Macoll", "x2" : "1-Strip MIR"}
for M_inf in Mach_Range:

    print(f'Mach: {str(M_inf)}')
    print(f'Beta\tExact Cone Angle\tCone Angle 1-Strip')

    # Start with minimum wave angle with round up avoiding division by 0
    Beta_swpt = np.arange(np.ceil(np.rad2deg(np.arcsin(1/M_inf))), 80)

    # List for Plot
    Cone_Angle_Exact_ls = [0]
    Cone_Angle_1strip_ls = [0]
    Beta_ls = [Beta_swpt[0]]

    for Beta in Beta_swpt:

        # Exact Solution
        Exact = TM(M_inf, gamma, Beta)
        _, Cone_Angle_Exact = TM.Taylor_Macoll_Solve(Exact)

        # One-Strip Solution
        MIR_1 = FCF(M_inf, gamma, Beta)
        Cone_Angle_1st = FCF.one_strip_solve(MIR_1, full_output=False)

        if Cone_Angle_Exact > Cone_Angle_Exact_ls[-1]:

            print(f'{Beta:4.2f}\t\t{Cone_Angle_Exact:7.4f}\t\t{Cone_Angle_1st:7.4f}')    
        
            Beta_ls.append(Beta)
            Cone_Angle_Exact_ls.append(Cone_Angle_Exact)
            Cone_Angle_1strip_ls.append(Cone_Angle_1st)

        else:
            break

    plt.plot(Cone_Angle_Exact_ls, Beta_ls, color='black', linewidth='1.5', label=my_labels["x1"])
    my_labels['x1'] = "_nolegend_"

    plt.plot(Cone_Angle_1strip_ls, Beta_ls, color='blue',linestyle='dashed', linewidth='1.5', label=my_labels["x2"])
    my_labels['x2'] = "_nolegend_"

plt.xlabel('Cone Angle \u03B8 [deg]')
plt.ylabel('Wave Angle \u03B2 [deg]')
plt.legend()
plt.grid()
plt.show()


