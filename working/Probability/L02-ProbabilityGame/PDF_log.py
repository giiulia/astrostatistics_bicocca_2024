import numpy as np
import statistics
import matplotlib.pyplot as plt

def f_y(y):
    return (10**y)*np.log(10)/(10 - 0.1)
    

N = 1000
x = np.random.uniform(0.1, 10, N)
mean_x = statistics.mean(x)
median_x = statistics.median(x)
plt.hist(x, bins = 20, edgecolor='black', alpha=0.7)

plt.xlabel('x')
plt.ylabel('Frequency')
plt.title("Histogram of uniformly distributed samples")
plt.show()
plt.savefig("Plots/PDF_uniform.png", format = "png", bbox_inches="tight")

y = np.log10(x)
mean_y = statistics.mean(y)
median_y = statistics.median(y)
mean_logx = np.log10(mean_x)
print(f"Mean of y: {mean_y} VS: ")
print(f"Mean of log(x): {mean_logx}")
median_logx = np.log10(median_x)
print(f"Median of y: {median_y} VS:")
print(f"Median of log(x): {median_logx}")

plt.hist(y, bins = 20, edgecolor='black', alpha=0.7, density = True)
ys = np.linspace(-1, 1) 
plt.plot(ys, f_y(ys))


plt.xlabel('y')
plt.ylabel('Frequency')
plt.title("Histogram of logaritmically distributed samples")
plt.show()
plt.savefig("Plots/PDF_log.png", format = "png", bbox_inches="tight")
