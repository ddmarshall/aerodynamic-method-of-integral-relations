import Method_of_Integral_Relations as MIR
from Method_of_Integral_Relations import Frozen_Blunt_Flow as FBF
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

print('\nPython: ' + sys.version)
print('Numpy Version: ' + np.__version__)

case1 = FBF(5, 1.4, 1)

# E0_esp0_fig = MIR.Blunt_E0_vs_esp0(Mach = [2.5, 3, 4, 5])
plt.show()


# Solve stand off distance
esp_0 = case1.One_Strip_Find_epsilon_0()

# Run full solution with given theta array for out put
theta_lis = np.arange(0, 1.125, 0.0625)
epsilon, sigma, v0, theta_sonic = case1.One_Strip_Solve_Full(esp_0, theta_lis)

MIR.plot_function(theta_lis, epsilon, sigma, v0)


plt.show()

pass