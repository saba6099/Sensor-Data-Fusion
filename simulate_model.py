# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:54:21 2019

@author: Tejas and Saba
"""

import numpy as np
import matplotlib.pyplot as plt


class DynamicsModel():
    def __init__(self, F, D):
        self.F = F
        self.D = D

        self.zero_mean = np.zeros(D.shape[0])

    def return_motion_model(self, x):
        new_state = self.F @ x + np.random.multivariate_normal(self.zero_mean, self.D)
        return new_state


class SensorModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        self.zero_mean = np.zeros(R.shape[0])

    def return_meas_model(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement

def simulate_system(K, x):
    (F, H, D, R) = model_parameters()
    dynamic_model = DynamicsModel(F, D)
    sensor_model = SensorModel(H, R)
    state = np.zeros((K, D.shape[0]))
    meas = np.zeros((K, R.shape[0]))

    for k in range(K):
        x = dynamic_model.return_motion_model(x)
        z = sensor_model.return_meas_model(x)

        state[k, :] = x
        meas[k, :] = z

    return state, meas

def model_parameters(T=2, s2_x=10, s2_y=10, lambda_sq=300):
    F = np.array([[1, T],
                  [0, 1]])
    base_sigma = np.array([[T ** 3 / 3, T ** 2 / 2],
                           [T ** 2 / 2, T]])

    sigma_x = s2_x * base_sigma
    sigma_y = s2_y * base_sigma

    zeros_2 = np.zeros((2, 2))
    F = np.block([[F, zeros_2],
                  [zeros_2, F]])
    D = np.block([[sigma_x, zeros_2],
                  [zeros_2, sigma_y]])

    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    R = lambda_sq * np.eye(2)

    return F, H, D, R


