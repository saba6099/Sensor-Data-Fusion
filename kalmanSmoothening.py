# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:59:32 2019

@author: Tejas and Saba
"""

import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from simulate_model import simulate_system, model_parameters
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg

def retrodiction(Xs, Ps, Fs=None, Qs=None):

    n = Xs.shape[0]
    dim_x = Xs.shape[1]
    Fs = [Fs] * n
    Qs = [Qs] * n

    K = zeros((n, dim_x, dim_x))
    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n - 2, -1, -1):
        pP[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]
        K[k] = dot(dot(P[k], Fs[k].T), linalg.inv(pP[k]))
        x[k] += dot(K[k], x[k + 1] - dot(Fs[k], x[k]))
        P[k] += dot(dot(K[k], P[k + 1] - pP[k]), K[k].T)

    return (x, P, K, pP)

np.random.seed(21)
# Initializing parameters
K = 20
v = 20
ax = 10
ay, az = 1, 1
t = (ax/v)
delta_t = 2

(F, H, Q, R) = model_parameters()
r_values = []
for t in range(K):

    r = [v*t, np.abs(ay*np.sin((4*np.pi*v*t)/ax)), np.abs(az*np.sin((np.pi*v*t)/ax))]
    r_values.append(r)

vel = [v, ay * ((4 * np.pi * v) / ax) * np.cos((4 * np.pi * v * 1) / ax)]
x = np.array((r_values[0][0], vel[0], r_values[0][1], vel[1]))
P = 0 * np.eye(4)
(state, meas) = simulate_system(K, x)
kalman_filter = KalmanFilter(F, H, Q, R, x, P)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))
cov = np.zeros((K,2,2))
for k in range(K):
    kalman_filter.predict()
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P
    c = np.asarray(([P[0][0:2],P[1][0:2], P[2][2:4], P[3][2:4]])[0:2])
    c = np.resize(c,(2,2))
    cov[k, ...] = c

mean = [est_state[:, 0], est_state[:, 2]]
mean = np.asarray(mean)
mean = np.transpose(mean)
F2 = np.asarray(([F[0][0:2], F[1][0:2], F[2][2:4], F[3][2:4]])[0:2])
F2 = np.resize(F2, (2, 2))
Q = 100
M,P,C,D = retrodiction(mean, cov, F2, Q)


plt.figure(figsize=(100, 50))
plt.plot(state[:, 0], state[:, 2], '-g')
plt.plot(est_state[:, 0], est_state[:, 2], '-ko')
plt.plot(meas[:, 0], meas[:, 1], ':rx')
plt.plot(M[:, 0], M[:, 1], 'b')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['True state', 'Kalman Filtered state', 'Measurement', 'Retrodiction'])
plt.axis('square')
plt.title('Kalman Filter with Retrodiction')
plt.plot()
plt.show()
