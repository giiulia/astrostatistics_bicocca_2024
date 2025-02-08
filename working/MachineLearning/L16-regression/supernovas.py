import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from astroML.linear_model import PolynomialRegression

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)

plt.show()

degree = 3
model = PolynomialRegression(degree) # fit 3rd degree polynomial

print(mu_sample.ndim)
mu_sample = mu_sample.reshape(-1, 1)
z_sample = z_sample.reshape(-1, 1)
model.fit(z_sample, mu_sample)


print(model.coef_)