import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

def kde_sklearn(data, bandwidth, kernel):
    kde_skl = KernelDensity(bandwidth = bandwidth, kernel=kernel)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sklearn returns log(density)

    return np.exp(log_pdf)


N = 1000
mu = 1
sigma = 0.02

M = norm(loc = 1, scale = sigma).rvs(N)
chi = np.random.uniform(0, 1, N)

Mirr = M*np.sqrt((1+np.sqrt(1-chi**2))/2)

a = 0.70
b = 1.05
Delta = 3.5*sigma/N**(1/3)
nbins = int((b-a)/Delta)
print(nbins)
plt.hist(Mirr, bins = nbins, density = True, label = 'data')

xgrid = np.linspace(Mirr.min(), Mirr.max(), 1000)

PDFgaussian = kde_sklearn(Mirr, bandwidth=0.01, kernel="gaussian") 
plt.plot(xgrid, PDFgaussian, label = 'KDE fit') 

plt.legend()
plt.show()

sigma_grid = np.logspace(0.001, 10, 100)

PDFgaussians = []
Ds1 = []
for s in sigma_grid:
    PDFgaussians.append( kde_sklearn(Mirr, bandwidth = s, kernel="gaussian") )
    difference = Mirr - PDFgaussians[-1]
    Ds1.append(difference.max())

plt.scatter(sigma_grid, Ds1)

plt.loglog()
plt.show()

Ds2 = []
for s in sigma_grid:
    difference = Mirr - M
    Ds2.append(difference.max())

plt.scatter(sigma_grid, Ds2)

plt.show()
