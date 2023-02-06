from Exact_Solution import Taylor_Macoll as TM
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4

Mach_Range = [1.5, 2, 2.6, 4, 10]

plt.figure(1)

for M_inf in Mach_Range:

    print(str(M_inf))

    Beta_swpt = np.arange(np.rad2deg(np.arcsin(1/M_inf)), 80)
    Cone_Angle_ls = [0]
    Beta_ls = [np.rad2deg(np.arcsin(1/M_inf))]

    for Beta in Beta_swpt:

        Flow = TM(M_inf, gamma, Beta)

        _, Cone_Angle = TM.Taylor_Macoll_Solve(Flow)

        if Cone_Angle > Cone_Angle_ls[-1]:

            print(Cone_Angle)    
            Cone_Angle_ls.append(Cone_Angle)
            Beta_ls.append(Beta)

        else:
            break

    plt.plot(Cone_Angle_ls, Beta_ls)

plt.grid()
plt.show()


