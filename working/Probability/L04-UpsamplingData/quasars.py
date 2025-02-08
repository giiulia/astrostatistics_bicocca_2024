from astroML.datasets import fetch_dr7_quasar
import matplotlib.pyplot as plt
import numpy as np
import astropy

# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]
z = data['redshift']
n, bins, patches = plt.hist(z, density=True, bins=100);
xcenters = (bins[:-1] + bins[1:]) / 2

plt.plot(xcenters, n)

plt.show()

#rejection sampling
N = 100000
x = np.random.uniform(0, xcenters[-1], N)
y = np.random.uniform(0, max(n), N) 

indexes = np.searchsorted(xcenters, x)
goodpoints = x[y < n[indexes]]

plt.hist(goodpoints, bins = 100, density=True, color='green')
plt.plot(xcenters, n)

plt.show()

#inverse sampling
cumulative = []
for a in range(1, 101):
    cumulative.append(sum(n[:a]))
cumulative = cumulative / cumulative[-1]

j = np.random.uniform(0, 1, N)
w = xcenters[np.searchsorted(cumulative, j)]
plt.hist(w, bins = 100, density=True, color='green')
plt.plot(xcenters, n/cumulative[-1])

plt.show()

#best fit
f = astropy.cosmology.Planck18
xg = np.linspace(0,7,100)
plt.plot(f)
plt.show()