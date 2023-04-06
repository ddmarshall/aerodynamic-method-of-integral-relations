import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
sigma = np.linspace(np.pi/2, 0.3)
w_inf = 0.8017837257372731
omega = ((w_inf**2)*(np.sin(sigma)**2))/(1 - w_inf**2)

phi1 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*(omega - (((gamma-1)**2)/(4*gamma)))*(1 + (1/omega))**gamma

domega_dsigma = ((w_inf**2)*(2*np.sin(sigma)*np.cos(sigma)))/(1 - w_inf**2)
d2omega_dsigma2 = 2*(w_inf**2)*np.cos(2*sigma)/(1 - w_inf**2)

dphi1_dsigma = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*((1 - (gamma/(omega+1)))*domega_dsigma*((1+(1/omega))**gamma) - ((gamma-1)**2)/(4*gamma)*(-gamma*(1/omega**2)*((1+(1/omega))**(gamma-1))*domega_dsigma))

d2phi1_dsigma2 = ((4*gamma)/((gamma**2)-1))*(((gamma-1)/(gamma+1))**gamma)*((((1+(1/omega))**gamma)*((gamma/(1+(1/omega)))*(-(omega**-2))*(domega_dsigma**2)+d2omega_dsigma2)-gamma*((1+(1/omega))**(gamma-1))*((-omega**-2)*(domega_dsigma**2)-(omega**-3)*((gamma-1)/(1+(1/omega)))*(domega_dsigma**2) + (omega**-1)*d2omega_dsigma2)) - ((((gamma-1)**2)/(4*gamma))*(-gamma*((1+(1/omega))**(gamma-1))*(-2*(omega**-3)*(domega_dsigma**2)-(omega**-4)*((gamma-1)/(1+(1/omega)))*(domega_dsigma**2)+(omega**-2)*((1+(1/omega))**(gamma-1))*d2omega_dsigma2))))

plt.figure(0)
plt.plot(sigma, phi1)
plt.plot(sigma, dphi1_dsigma)
plt.plot(sigma, d2phi1_dsigma2)
plt.xlabel('Sigma')
plt.ylabel('phi1')
plt.legend(['phi_1','dphi_1/dsigma','d2phi_1/dsigma2'])
plt.grid()
plt.show()