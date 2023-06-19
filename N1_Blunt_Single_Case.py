import Method_of_Integral_Relations as MIR
import matplotlib.pyplot as plt

# Define Case Parameters
case = MIR.Frozen_Blunt_Flow(M_inf=3, gamma=1.4, N=1, cutoff=0.95, print_sol=True)

# Run the Solver
[epsilon, sigma, v0, theta_v0s, theta_w1s, theta_lis] = MIR.One_Strip_Single_Case(case)

# Input Variable names to plot
resutls_figures = case.plot_properties('epsilon','v0','p0/p0(0)','kai','w1')

# Add data from Dr.B and Kim
case.plot_compare(resutls_figures, Mach=case.M_inf, N=3)

# Contour Plots, come with comaprations
case.plot_contour(epsilon, sigma, v0, theta_lis[theta_lis.index(theta_v0s)], theta_lis[theta_lis.index(theta_w1s)])
 

plt.show()
    
