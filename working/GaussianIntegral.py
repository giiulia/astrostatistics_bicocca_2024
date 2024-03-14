import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f(x):
    return x**3

def sin_func(x, A, B, C):
    y = A + B*np.sin(C/x)
    return y

sigma = 2
normalization = np.sqrt(2*np.pi*sigma**2)

integrals = []
errors = []
solution = 2*sigma**4

N = range(10, 1000, 10)
for i in N:
    xj = np.abs(np.random.normal(0, sigma, i))

    integral = normalization/2 *np.mean(f(xj))
    error = normalization/2 * np.std(f(xj))/np.sqrt(i)
   # print(f"integral {integral} sol: {solution}")
    integrals.append(integral)
    errors.append(error)

#trend as N varies
plt.errorbar(N, integrals, errors, label = 'integrals')
plt.axhline(y = 2*sigma**4, color = 'g', linestyle = '--', label = '$2 \sigma^4$')
plt.xlabel("N")

parameters, covariance = curve_fit(sin_func, integrals, errors)
fit_A = 32
fit_B = parameters[1]
fit_C = parameters[2]

fit = sin_func(integrals, fit_A, fit_B, fit_C)

plt.plot(N, fit, '-', label='fit')


plt.legend()
plt.show()
