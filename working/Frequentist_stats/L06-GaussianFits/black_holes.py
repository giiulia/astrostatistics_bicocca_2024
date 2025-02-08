import numpy as np
import sklearn.mixture 
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

seed = np.random.seed(1)

data = np.load('formationchannels.npy')
plt.hist(data, label = 'data', bins = 100, density = True)

gm_models = np.array([sklearn.mixture.GaussianMixture(n, random_state = seed).fit(data) for n in range(1, 11)])

x = np.linspace(-20, 80, 100).reshape(-1, 1) #-1 means numpy figures out the number of rows alone in order to be compatible with before

for n in range(10): #score samples Compute the log-likelihood of each sample under the nth model. (x must be array-like of shape (n_samples, n_features))
    logprob = gm_models[n].score_samples(x) 
    pdf = np.exp(logprob)
    plt.plot(x, pdf, label = f'{n+1} Gauss')

plt.legend()
plt.show()

print(dir(gm_models[0]))
AICs = np.array([ gm_models[n-1].aic(data)  for n in range(1, 11)])
ns = np.arange(1, 11)
plt.plot(ns, AICs, ls = '-')
plt.show()