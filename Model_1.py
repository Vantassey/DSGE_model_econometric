import numpy as np

# Parameters
alpha = 0.33
beta = 0.96
delta = 0.05
rho = 0.9
sigma = 0.01

# Steady state values
k_ss = (alpha / (1/beta - 1 + delta))**(1 / (1 - alpha))
c_ss = k_ss**alpha - delta * k_ss

# Simulation
T = 100
np.random.seed(123)
epsilon = np.random.normal(0, sigma, T)

k = np.zeros(T+1)
c = np.zeros(T)

k[0] = k_ss

for t in range(T):
    c[t] = k[t]**alpha - k[t+1] + (1 - delta) * k[t]
    k[t+1] = alpha * beta * (k[t]**alpha) + (1 - delta) * k[t] + epsilon[t]

# Plotting
import matplotlib.pyplot as plt

plt.plot(range(T), c, label='Consumption')
plt.plot(range(T+1), k, label='Capital')
plt.axhline(y=c_ss, color='r', linestyle='--', label='Steady state consumption')
plt.axhline(y=k_ss, color='g', linestyle='--', label='Steady state capital')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('DSGE Model Simulation')
plt.show()
