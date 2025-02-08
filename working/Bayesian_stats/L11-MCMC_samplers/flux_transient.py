import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

def model(t, param):
    b, t0, A, alpha = param
    return np.where(t<t0, b, b+A*np.exp(-alpha*(t-t0)))   

def LogLikelihood(x, times, flux, err_flux): #assumption: Likelihood has a gaussian shape (== the process IS gaussian)
    y_fit = [model(t, x) for t in times]
    return np.sum(-0.5*((flux - y_fit)**2/err_flux**2))

def LogPrior(x): #up to a constant
    b, t0, A, alpha = x
    if 0.0 < b < 50.0 and 0.0 < t0 < 100.0 and 0.0 < A < 50.0 and np.exp(-5) < alpha < np.exp(5):
        return 0.0 + np.log(1/alpha)
    else:
        return -np.inf

def LogPosterior(x, data):
    times, flux, err_flux = data
    return LogLikelihood(x, times, flux, err_flux) + LogPrior(x)

transient_data = np.load('flux_transient.npy')
print(transient_data)

time, flux, err_flux = transient_data.T
print(transient_data.T)

plt.xlabel("Time")
plt.ylabel("Flux")

plt.errorbar(time, flux, xerr = None, yerr = err_flux, marker='s', mec='red', ms=2, mew=2)

parameters = [10, 50, 5, 0.1] #b, t0, A, alpha

xs = np.linspace(1, 100, 1000)
ys = [model(x, parameters) for x in xs]
plt.plot(xs, ys)
plt.show()

#MCMC
ndim = 4
nwalkers = 20
nsteps = 10000


np.random.seed(0)
p0 = parameters + 1e-1 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior, args = [transient_data.T])
sampler.run_mcmc(p0, nsteps)

print("done")

fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
labels = ["b","t0","A","alpha"]

samples = sampler.get_chain()
print(samples.shape)

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha = 0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

plt.show()

#thinning
tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True) #flat = all 20 walkers together: from 10000 down to 2260
print(flat_samples.shape)

#corner plot
fig = corner.corner(flat_samples, labels=labels, levels=[0.68, 0.95])
plt.show()

#plot N random posteriors

N = 100 #chosen models to plot
chosen_samples = flat_samples[np.random.choice(len(flat_samples), size = N)]

plt.xlabel("Time")
plt.ylabel("Flux")

plt.errorbar(time, flux, xerr = None, yerr = err_flux, marker='s', mec='red', ms=2, mew=2)
parameters = chosen_samples

xs = np.linspace(1, 100, 1000)
for i in range(N):
    ys = [model(x, parameters[i]) for x in xs]
    plt.plot(xs, ys, alpha = 0.2)

plt.show()

#summary stats
for i,l in enumerate(labels):
    low, median, up = np.percentile(flat_samples[:,i],[5,50,95]) 
    print(l+"   "+str(median)+" +"+str(up - median)+" -"+str(median - low))
    
    