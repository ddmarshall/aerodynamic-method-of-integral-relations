from TM_Exact_Solution import Taylor_Macoll as TM
from Method_of_Integral_Relations import Frozen_Cone_Flow as FCF
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4

Mach_Range = [1.5, 2, 2.5, 4, 10, 15]

plt.figure(1)

my_labels = {"x1" : "Taylor-Macoll", "x2" : "1-Strip MIR", "x3" : "2-Strip MIR"}
for M_inf in Mach_Range:

    print(f'Mach: {str(M_inf)}')
    print(f'Beta\tExact Cone Angle\tCone Angle 1-Strip\tCone Angle 2-Strip\t2-Strip Solution Msg:')
    
    # Start with minimum wave angle with round up avoiding division by 0
    Beta_swpt = np.arange(np.ceil(np.rad2deg(np.arcsin(1/M_inf))), 80)

    # List for Plot
    Cone_Angle_Exact_ls = [0]
    Cone_Angle_1strip_ls = [0]
    Cone_Angle_2strip_ls = []
    Beta_ls = [Beta_swpt[0]]
    Beta_2strip_ls = []

    for Beta in Beta_swpt:

        # Exact Solution
        Exact = TM(M_inf, gamma, Beta)
        _, Cone_Angle_Exact = TM.Taylor_Macoll_Solve(Exact)

        # One-Strip Solution
        MIR_1 = FCF(M_inf, gamma, Beta)
        Cone_Angle_1st = FCF.one_strip_solve(MIR_1, full_output=False)
        
        # Two-Strip Solution
        Cone_Angle_2nd, solver_msg = FCF.two_strip_solve(MIR_1, full_output=False)

        if Cone_Angle_Exact > Cone_Angle_Exact_ls[-1]:

            print(f'{Beta:4.2f}\t\t{Cone_Angle_Exact:7.4f}\t\t{Cone_Angle_1st:7.4f}\t\t{Cone_Angle_2nd:7.4f}\t\t{solver_msg}')
        
            Beta_ls.append(Beta)
            Cone_Angle_Exact_ls.append(Cone_Angle_Exact)
            Cone_Angle_1strip_ls.append(Cone_Angle_1st)

            if solver_msg == 'Converged.' and Cone_Angle_2nd < Beta:
                Beta_2strip_ls.append(Beta)
                Cone_Angle_2strip_ls.append(Cone_Angle_2nd)

        else:
            break

    plt.plot(Cone_Angle_Exact_ls, Beta_ls, color='black', linewidth='1.5', label=my_labels["x1"])
    my_labels['x1'] = "_nolegend_"

    plt.plot(Cone_Angle_1strip_ls, Beta_ls, color='blue',linestyle='dashed', linewidth='1.5', label=my_labels["x2"])
    my_labels['x2'] = "_nolegend_"

    plt.plot(Cone_Angle_2strip_ls, Beta_2strip_ls, color='red',linestyle='dashed', linewidth='1.5', label=my_labels["x3"])
    my_labels['x3'] = "_nolegend_"

    plt.text(-2.75, Beta_ls[0], f'Mach {M_inf}', fontsize=20)

plt.xlabel('Cone Angle \u03B8 [deg]', fontsize=20)
plt.ylabel('Wave Angle \u03B2 [deg]', fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()


