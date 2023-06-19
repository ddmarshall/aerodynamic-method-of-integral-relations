import json as js
from matplotlib import pyplot as plt
  
# Opening JSON file
shock_geo_OMB1958 = js.load(open('data/M3_epsilon_N1-3_OMB1958.json'))
N1_M3_shock_geo_95cut = js.load(open('data/N1_M3_shock_geo_cut95_result.json'))
N1_M3_shock_geo_90cut = js.load(open('data/N1_M3_shock_geo_cut90_result.json'))


plt.figure()
plt.plot(shock_geo_OMB1958['N=1']['theta'], shock_geo_OMB1958['N=1']['epsilon'],'--o', label='OMB1959, N=1',color='black',linewidth=0.5)
plt.plot(N1_M3_shock_geo_95cut['theta'], N1_M3_shock_geo_95cut['epsilon'], '--*', label='Cut95%, N=1',color='black',linewidth=0.5)
plt.plot(N1_M3_shock_geo_90cut['theta'], N1_M3_shock_geo_90cut['epsilon'], '--^', label='Cut90%, N=1',color='black',linewidth=0.5)
plt.ylabel('\u03B5')
plt.xlabel('\u03B8 [rad]')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(shock_geo_OMB1958['N=1']['theta'], shock_geo_OMB1958['N=1']['p0/p0(0)'],'--o', label='OMB1959, N=1',color='black',linewidth=0.5)
plt.plot(N1_M3_shock_geo_95cut['theta'], N1_M3_shock_geo_95cut['p0/p0(0)'], '--*', label='Cut95%, N=1',color='black',linewidth=0.5)
plt.plot(N1_M3_shock_geo_90cut['theta'], N1_M3_shock_geo_90cut['p0/p0(0)'], '--^', label='Cut90%, N=1',color='black',linewidth=0.5)
plt.ylabel('p0/p0(0)')
plt.xlabel('\u03B8 [rad]')
plt.legend()
plt.grid()
plt.show()