import numpy as np
import pylab as plt
import statistics as stat
import scipy.stats

def f(x):
    return x**3


#calculate integral
def integral(sigma, samples):
    xi = np.abs(gauss.rvs(samples))
    integral = sigma*(1/2)*(np.pi*2)**0.5 * np.mean(xi**3)
    return integral

def error(sigma, samples):
    xi = np.abs(gauss.rvs(samples))
    error = sigma*(1/2)*(np.pi*2)**0.5 * np.std(xi**3) * samples**(-0.5)
    return error

#gaussian distribution from 0 to 3sigma
sigma = 0.45
samples = 1000
gauss = scipy.stats.norm(loc=0,scale=sigma)
plt.hist(np.abs(gauss.rvs(samples)),density=True,bins=30)
plt.plot(np.linspace(0,3*sigma,100), 2*gauss.pdf(np.linspace(0,3*sigma,100)))

plt.show()

#calculate integral multiple times at different number of samples
solution = 2*sigma**4

Nsamples = np.unique(np.logspace(0, 6, 100, dtype=int))
integrals = np.array([integral(sigma, i) for i in Nsamples])
errors = np.array([error(sigma, i) for i in Nsamples])

plt.axhline(solution, c='red')
plt.plot(Nsamples, integrals)
plt.loglog()
plt.show()

plt.axhline(0, c = 'red')
plt.plot(Nsamples, errors)
plt.plot(Nsamples, Nsamples**(-0.5))
plt.loglog()
plt.show()

