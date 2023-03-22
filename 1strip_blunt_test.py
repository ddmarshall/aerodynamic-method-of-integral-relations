from Method_of_Integral_Relations import Frozen_Blunt_Flow as FBF
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

print('\nPython: ' + sys.version)
print('Numpy Version: ' + np.__version__)

case1 = FBF(3, 1.4, 1)
# Solve stand off distance
esp_0 = case1.One_Strip_Find_epsilon_0()

[Flow_Solution, _] = case1.One_Strip_Solve_Sonic(esp_0)

# Extract Dense Output
theta = np.linspace(Flow_Solution.t.min(), Flow_Solution.t.max(),70)
epsilon = Flow_Solution.sol(theta)[0]
sigma = Flow_Solution.sol(theta)[1]
v_0 = Flow_Solution.sol(theta)[2]

# Create E_0 for plot
E_0_lis = [case1.E(0, theta[i], sigma[i], epsilon[i]) for i in range(len(theta))]


plt.figure()
plt.subplot(4,1,1)
plt.plot(theta, E_0_lis)
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