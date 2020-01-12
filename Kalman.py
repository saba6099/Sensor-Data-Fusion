# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:59:32 2019

@author: Tejas
"""



import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from simulate_model import simulate_system, create_model_parameters

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

z1 = np.zeros(K)
z2 = np.zeros(K)

x_s1 = 0
x_s2 = 100
y_s1 = 100
y_s2 = 0
z_s1 = z_s2 = 10
sensor1 = []
sensor2 = []
sensor1_coor = []


for i in range(K):
    range = np.sqrt(np.square(r_values[i][0] - x_s1) + np.square(r_values[i][1] - y_s1) + np.square(r_values[i][2] - z_s1) - np.square(z_s1)) + 10*np.random.normal(0,1)
    azimuth = np.arctan((r_values[i][1] - y_s1)/ (r_values[i][0] - x_s1)) + 0.1*np.random.normal(0,1)
    sensor1.append([range, azimuth])
    vel = [v, ay * ((4 * np.pi * v) / ax) * np.cos((4 * np.pi * v * i) / ax)]
    val = [range * np.cos(azimuth), vel[0], range*np.sin(azimuth), vel[1]]
    sensor1_coor.append(val)

    range = np.sqrt(np.abs(np.square(r_values[i][0] - x_s2) + np.square(r_values[i][1] - y_s2) + np.square(r_values[i][2] - z_s2) - np.square(z_s2))) + 10*np.random.normal(0,1)
    azimuth = np.arctan((r_values[i][1] - y_s2) / (r_values[i][0] - x_s2)) + 0.1*np.random.normal(0,1)
    sensor2.append([range, azimuth])

vel = [v, ay * ((4 * np.pi * v) / ax) * np.cos((4 * np.pi * v * 1) / ax)]
# for i in range(899):
#     val = sensor1[i][0]*np.cos(sensor1[i][1])#, sensor1[i][0]*np.sin(sensor1[i][1])]
#     sensor1_coor.append(val)



x = np.array((r[0], vel[0], r[1], vel[1]))
x = sensor1_coor[0]
# x = [100,100.5,5000,1000.5]
np.random.seed(21)
(A, H, Q, R) = create_model_parameters()
# K = 20
# initial state

# x = np.array([0, 0.1, 0, 0.1])
P = 0 * np.eye(4)
block = np.array(([62500, 0], [0, 40000]))
zeros_2 = np.zeros((2, 2))
P = np.block([[R, zeros_2],
                  [zeros_2, block]])
(state, meas) = simulate_system(K, x, sensor1_coor)
kalman_filter = KalmanFilter(A, H, Q, R, x, P)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))
d = 0
while d < K :
    kalman_filter.predict()
    kalman_filter.update(meas[d, :])
    (x, P) = kalman_filter.get_state()

    est_state[d, :] = x
    est_cov[d, ...] = P
    d = d+1

plt.figure(figsize=(7, 5))
plt.plot(state[:, 0], state[:, 2], '-bo')
plt.plot(np.asarray(r_values)[:,0], np.asarray(r_values)[:,1], '-y')
plt.plot(np.asarray(sensor1_coor)[:,0], np.asarray(sensor1_coor)[:,1], '-g')
plt.plot(est_state[:, 0], est_state[:, 2], '-ko')
plt.plot(meas[:, 0], meas[:, 1], ':rx')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state','original','sensor', 'inferred state', 'observed measurement'])
plt.axis('square')
plt.tight_layout(pad=0)
plt.show()
plt.plot()
