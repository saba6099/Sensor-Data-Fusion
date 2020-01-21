# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:58:01 2019

@author: Tejas
@author: Saba
"""
import numpy as np


class KalmanFilter():
    def __init__(self, F, H, D, R, x_0, P_0):
        self.F = F
        self.H = H
        self.D = D
        self.R = R

        self._x = x_0
        self._P = P_0

    def predict(self):
        self._x = self.F @ self._x
        self._P = self.F @ self._P @ self.F.transpose() + self.D

    def update(self, z):
        self.S = self.H @ self._P @ self.H.transpose() + self.R
        self.V = z - self.H @ self._x
        self.K = self._P @ self.H.transpose() @ np.linalg.inv(self.S)

        self._x = self._x + self.K @ self.V
        self._P = self._P - self.K @ self.S @ self.K.transpose()

    def get_state(self):
        return self._x, self._P
