import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy.optimize import curve_fit
def f(x):
    return x**3


def Gauss(x, A, B, C):
    y = A*np.exp(-((x-B)**2)/C**2)
    return y

sigma = 2
samples = 10000

integrals = []
solution = 2*sigma**4

N = 1000
for i in range(N):
    xi = np.random.normal(0, sigma, samples)

    integral = np.sqrt(2*np.pi*sigma**2)/2 *np.sum(abs(f(xi)))/samples
   # print(f"integral {integral} sol: {solution}")
    integrals.append(integral)

h, xedges, patches = plt.hist(integrals, density=True, bins=20, label=f"mean: {stat.mean(integrals):.2f}, std: {stat.pstdev(integrals):.2f}")
xcenters = (xedges[:-1] + xedges[1:]) / 2

parameters, covariance = curve_fit(Gauss, xcenters, h)
fit_B = sum(integrals)/len(integrals)
variance = sum([((x - fit_B) ** 2) for x in integrals]) / len(integrals) 
fit_C = variance ** 0.5
fit_A = 1/np.sqrt(2*np.pi*variance)

print(fit_A)
print(fit_B)
print(fit_C)
fit_y = Gauss(xcenters, fit_A, fit_B, fit_C)
plt.plot(xcenters, h, 'o', label='data')
plt.plot(xcenters, fit_y, '-', label='fit')

plt.legend(loc="upper left")
plt.ylabel("Frequency")
plt.xlabel("Integrals")
plt.legend()
plt.show()
