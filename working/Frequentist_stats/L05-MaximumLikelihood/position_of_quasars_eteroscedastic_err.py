import numpy as np
import pylab as plt
from scipy.stats import norm

np.random.seed(42)
N = 5

mean = 1
sigma = 0.2
sigma_err = 0.05

err_gauss_dist = norm(loc = sigma, scale = sigma_err)
err_data = err_gauss_dist.rvs(N)

gauss_data = np.concatenate([norm(loc = mean, scale = s).rvs(1) for s in err_data])

average = np.average(gauss_data, weights = 1/err_data**2)
average_err = np.sum(err_data**-2)**-0.5

plt.axvline(average, c='red', label = 'average')

mu_proposed = np.linspace(mean-3*sigma, mean+3*sigma, 1000)

Lsamples = np.array([norm.pdf(mu_proposed, loc = m, scale = s) for m,s in zip (gauss_data,err_data)]) 

for L in Lsamples:
    plt.plot(mu_proposed, L, ls = '-', label = 'sample')

L = np.prod(Lsamples, axis = 0)

plt.plot(mu_proposed, L, label = 'Likelihood', ls = '--')
plt.xlabel('$\mu$')
plt.ylabel('$p(x_i|\mu, \sigma)$')

i_max = (np.argsort(L))[-1]
mu_max = mu_proposed[i_max]

plt.axvline(mu_max, c='red', label = 'X_maximum')

plt.legend()
plt.show()

print(f"average: {average}, max_lnL: {mu_max}")

#check Fisher matrix error
lnL_diff = np.diff(np.log(L), 2)
parabola_coeff = lnL_diff[i_max]/(mu_proposed[1] - mu_proposed[0])**2
estimate_error = (-parabola_coeff)**-0.5
print(f"average error: {average_err}, estimate_error: {estimate_error}")

#check goodness of estimators
Lfit = norm.pdf(mu_proposed, loc = mu_max, scale = estimate_error)
C = 1.42
plt.plot(mu_proposed, Lfit*C, ls = '-', label = '$L_{fit}$')
plt.plot(mu_proposed, L, ls = '--', label = 'L')
plt.legend()
plt.show()
