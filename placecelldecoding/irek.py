# -*- coding: utf-8 -*-
'''
Place cell decoding using Iterated Extended Kalman Filter

Author: Shashwat Shukla
Date: 23rd November 2018
'''
# Import libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Function to compute the log-posterior
def posterior(x, y, param, xhat, Wihat):
    # Compute lambda
    lam = np.exp(- 0.5 * (1.0 * (x[0] - param[:, 0]) / param[:, 2]) ** 2)
    lam = lam * np.exp(- 0.5 * (1.0 * (x[1] - param[:, 1]) / param[:, 3]) ** 2)
    lam = lam * np.exp(param[:, 4])
    log_lam = - 0.5 * (1.0 * (x[0] - param[:, 0]) / param[:, 2]) ** 2
    log_lam += - 0.5 * (1.0 * (x[1] - param[:, 1]) / param[:, 3]) ** 2
    log_lam += param[:, 4]
    # Compute negative log-posterior
    logl = np.sum(-y * log_lam + lam) + 0.5 * \
        np.dot((x - xhat), np.matmul(Wihat, (x - xhat)))
    return logl


# Function to compute the derivative of the log-posterior function
def der(x, y, param, xhat, Wihat):
    # Compute lambda
    lam = np.exp(- 0.5 * (1.0 * (x[0] - param[:, 0]) / param[:, 2]) ** 2)
    lam = lam * np.exp(- 0.5 * (1.0 * (x[1] - param[:, 1]) / param[:, 3]) ** 2)
    lam = lam * np.exp(param[:, 4])
    # Compute derivatives of log-posterior
    dl1 = -1.0 * (x[0] - param[:, 0]) / param[:, 2] ** 2
    dl2 = -1.0 * (x[1] - param[:, 1]) / param[:, 3] ** 2
    dlogl = np.zeros(2)
    dlogl[0] = np.sum(-y * dl1 + lam * dl1)
    dlogl[1] = np.sum(-y * dl2 + lam * dl2)
    dlogl = dlogl + np.matmul(Wihat, (x - xhat))
    return dlogl


# Function to compute the Hessian of the log-posterior function
def hess(x, y, param, xhat, Wihat):
    # Compute lambda
    lam = np.exp(- 0.5 * (1.0 * (x[0] - param[:, 0]) / param[:, 2]) ** 2)
    lam = lam * np.exp(- 0.5 * (1.0 * (x[1] - param[:, 1]) / param[:, 3]) ** 2)
    lam = lam * np.exp(param[:, 4])
    # Compute derivatives of log-posterior
    dl1 = -1.0 * (x[0] - param[:, 0]) / param[:, 2] ** 2
    dl2 = -1.0 * (x[1] - param[:, 1]) / param[:, 3] ** 2
    dlogl = np.zeros(2)
    dlogl[0] = np.sum(-y * dl1 + lam * dl1)
    dlogl[1] = np.sum(-y * dl2 + lam * dl2)
    dlogl = dlogl + np.matmul(Wihat, (x - xhat))
    # Compute Hessian of log-posterior
    H = np.zeros((2, 2))
    H[0, 0] = np.sum(y / param[:, 2] ** 2 + lam *
                     dl1 * dl1 - lam / param[:, 2] ** 2)
    H[0, 1] = np.sum(lam * dl1 * dl2)
    H[1, 0] = H[0, 1]
    H[1, 1] = np.sum(y / param[:, 3] ** 2 + lam *
                     dl2 * dl2 - lam / param[:, 3] ** 2)
    H = H + Wihat
    return H


# Seed the random number generator with the current time
np.random.seed(np.int(time.time()))

# Simulate random walk
T = 2000  # number of time-steps
path = np.random.randn(T, 2) * 0.03
path = np.cumsum(path, axis=0)

# Create place cells
N = 25  # Number of place cells
param = np.zeros((N, 5))  # Parameter table for the place cells
param[:, 0:2] = np.random.randn(N, 2)  # Centres of the receptive fields
param[:, 2:4] = np.abs(np.random.randn(N, 2))  # Size of the receptive fields
param[:, 4] = np.random.randn(N) * 0.5  # Offsets of the receptive fields

# Compute responses of place cells along the traversed path
R = np.zeros((T, N))  # Spiking responses of the place cells
spikes = [[] for i in range(N)]  # Raster plot
for i in range(N):
    r = np.exp(- 0.5 * ((path[:, 0] - param[i, 0]) / param[i, 2]) ** 2)
    r = r * np.exp(- 0.5 * ((path[:, 1] - param[i, 1]) / param[i, 3]) ** 2)
    r = r * np.exp(param[i, 4])
    r = np.random.poisson(r)
    R[:, i] = r
    spikes[i] = np.where(r == 1)[0]

# Display raster plot
if(len(spikes[0]) == 0):  # Handle bug in eventplot
    spikes[0] = np.array(1)
plt.eventplot(spikes)
plt.title('Raster plot', fontweight='bold')
plt.ylabel('Neuron index', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Decoding stage
W = np.identity(2) * 0.03 ** 2
xhat = np.zeros(2)
What = np.identity(2) * 5
xs = np.zeros((T, 2))
Ws = np.zeros((T, 2, 2))

for i in range(T):
    What = What + W
    y = R[i]
    Wihat = np.linalg.inv(What)
    est = minimize(posterior, xhat, args=(y, param, xhat, Wihat),
                   method='Newton-CG', jac=der, hess=hess, options={'xtol': 1e-4, 'disp': False})
    xhat = est.x
    Wihat = hess(xhat, y, param, xhat, Wihat)
    What = np.linalg.inv(Wihat)
    xs[i] = xhat
    Ws[i] = What

# Display recorded and decoded path of rat
plt.figure()
plt.plot(path[:, 0], path[:, 1] , label='Actual path')
plt.plot(xs[:, 0], xs[:, 1], label='Decoded path')
plt.legend()
plt.title('Path of the rat', fontweight='bold')
plt.show()

# Display recorded and decoded x-coordinate
plt.figure()
plt.plot(path[:, 0], label='True')
plt.plot(xs[:, 0], label='Decoded')
plt.legend()
plt.title('x-coordinate', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()

# Display recorded and decoded y-coordinate
plt.figure()
plt.plot(path[:, 0], label='True')
plt.plot(xs[:, 0], label='Decoded' )
plt.legend()
plt.title('y-coordinate', fontweight='bold')
plt.xlabel('Time (msec)', fontweight='bold')
plt.show()