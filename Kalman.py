# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:59:32 2019

@author: Tejas
"""



import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from simulate_model import simulate_system, create_model_parameters

np.random.seed(21)
(A, H, Q, R) = model_parameters()
K = 20
# initial state
# x = np.array([0, 0.1, 0, 0.1])
v = 20
ax = 10
ay, az = 1, 1
t = (ax/v)
delta_t = 2
K = 20

t = 1
# r = [v*t, ay*np.sin((4*np.pi*v*t)/ax), az*np.sin((np.pi*v*t)/ax)]
# vel = [v, ay*((4*np.pi*v)/ax)*np.cos((4*np.pi*v*t)/ax), az*((np.pi*v)/ax)*np.cos((np.pi*v*t)/ax)]
x_k = np.zeros((K, 3))
r_values = []
for t in range(K):

    r = [v*t, np.abs(ay*np.sin((4*np.pi*v*t)/ax)), np.abs(az*np.sin((np.pi*v*t)/ax))]
    r_values.append(r)
# P = 0 * np.eye(4)
vel = [v, ay * ((4 * np.pi * v) / ax) * np.cos((4 * np.pi * v * 1) / ax)]
x = np.array((r_values[0][0], vel[0], r_values[0][1], vel[1]))
P = 0 * np.eye(4)
(state, meas) = simulate_system(K, x)
kalman_filter = KalmanFilter(A, H, Q, R, x, P)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))

for k in range(K):
    kalman_filter.predict()
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P

mean = [est_state[:, 0], est_state[:, 2]]
cov = [est_cov[:, 0], est_cov[:, 2]]


plt.figure(figsize=(7, 5))
plt.plot(state[:, 0], state[:, 2], '-bo')
plt.plot(est_state[:, 0], est_state[:, 2], '-ko')
plt.plot(meas[:, 0], meas[:, 1], ':rx')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state', 'inferred state', 'observed measurement'])
plt.axis('square')
plt.tight_layout(pad=0)
plt.plot()
plt.show()