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
# for i in range(len(case1.theta_list)):
    # if case1.theta_list[i] == theta:
        # E_0_lis.append(case1.E0_list[i])


plt.figure()
plt.subplot(4,1,1)
plt.plot(case1.E0_list)
plt.ylabel('E_0')
plt.grid()
plt.subplot(4,1,2)
plt.plot(theta, v_0)
plt.ylabel('v_0')
plt.grid()
plt.subplot(4,1,3)
plt.plot(theta, epsilon)
plt.ylabel('\u03B5')
plt.grid()
plt.subplot(4,1,4)
plt.plot(theta, sigma)
plt.ylabel('\u03C3')
plt.grid()
plt.show()

print(' ')