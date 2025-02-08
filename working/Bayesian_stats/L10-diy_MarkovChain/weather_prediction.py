import matplotlib.pyplot as plt
import numpy as np
import random
import astroML.stats

N = 10000
forecast = []
# 0 = Cloudy
# 1 = Sunny
forecast.append(0)

for i in range(N):
    if forecast[-1] == 0:
        t = random.randint(0, 1)
        forecast.append(t)
    else:
        t = random.randint(0, 9)
        if t > 0:
            forecast.append(1)
        else:
            forecast.append(0)

forecast = np.array(forecast)
print(forecast)
plt.hist(forecast, density = True, bins = 100)
plt.show()

clear_days = np.cumsum(forecast)/(np.arange(len(forecast)) + 1)
plt.plot(clear_days)
plt.show()

plt.hist(clear_days, bins=300, density=True)
plt.show()

print(f"median: {np.median(clear_days)}")
print(f"sigma: {astroML.stats.sigmaG(clear_days)}")
print(f"min: {min(clear_days)}, max: {max(clear_days)}")

burn_in = 2000
clear_days_burn = clear_days[burn_in:]

plt.hist(clear_days_burn, bins=300, density=True)
plt.show()
print(np.median(clear_days_burn))
print(astroML.stats.sigmaG(clear_days_burn))
print(min(clear_days_burn), max(clear_days_burn))
