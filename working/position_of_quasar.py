import numpy as np
import pylab as plt
import scipy.stats
from scipy.stats import norm

mean = 1
sigma = 0.2
gauss_dist = norm(loc = 1, scale = sigma)

gauss_data = gauss_dist.rvs(100)
average = np.mean(gauss_data)
error_avg = np.std(gauss_data)

plt.plot(np.linspace(mean-3*sigma, mean+3*sigma, 100), gauss_dist.pdf(np.linspace(mean-3*sigma, mean+3*sigma, 100)), label = 'pdf')
plt.axvline(average, c='red', label = 'average')
#plt.show()

mu_proposed = np.linspace(-20, 20, 100) #40/100 = Delta theta 
lnL_scan = []
for mu in mu_proposed:
    #lnL_tmp = gauss_dist.logpmf(gauss_data)
    lnL_tmp = ((2*np.pi*sigma**2)**-0.5) * np.exp(-(gauss_data - mu)**2/2*sigma**2)
    lnL_tmp = np.sum(lnL_tmp) #product of logs
    lnL_scan.append( lnL_tmp )

i_max = lnL_scan.index(max(lnL_scan))
mu_max = mu_proposed[i_max]

lnL_scan = np.array(lnL_scan)
plt.plot(mu_proposed, lnL_scan)
plt.axvline(mu_max, c='red', label = 'X_maximum')
#plt.show()

print(f"average: {average}, max_lnL: {mu_max}")

#check Fisher
lnL_diff = np.diff(np.diff(lnL_scan))
parabola_coeff = lnL_diff[i_max]/(0.4)**2
estimate_error = (-parabola_coeff)**0.5
print(f"average error: {error_avg}, estimate_error: {estimate_error}")



