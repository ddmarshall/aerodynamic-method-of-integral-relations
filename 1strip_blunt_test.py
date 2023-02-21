from Method_of_Integral_Relations import Frozen_Blunt_Flow as FBF
import numpy as np
import matplotlib.pyplot as plt



case1 = FBF(3, 1.4, 1)

Flow_Solution = case1.One_Strip_Solve()

theta = Flow_Solution.t
epsilon = Flow_Solution.y[0]
sigma = Flow_Solution.y[1]
v_0 = Flow_Solution.y[2]

E_0_lis = []
for t in theta:
    E_0_lis.append(FBF.E(case1, 0, t))


plt.figure()
plt.plot(theta, E_0_lis)
plt.plot(theta, v_0)
plt.grid()
plt.show()

print(' ')