# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:22:42 2016

@author: Marko
"""

import numpy as np
from constants import *

def generateInput(idi, N, scale):  
    inputSpace = []
    if idi == RANDOM_ID:
        print(' | Random', end="")
        real = np.random.rand(N, N, N)
        real = 2 * real * scale - scale
        imag = np.random.rand(N, N, N)
        imag = 2 * imag * scale - scale
        inputSpace = real + 1j * imag
    elif idi == POINT_ID:
        print(' | Point', end="")
        h = 1.0 / N
        real = np.zeros((N,)*3)
        imag = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        real[mid][mid][mid] = scale / (h ** 3)
        inputSpace = real + 1j * imag
    elif idi == STICK_ID:
        print(' | Stick', end="")
        h = 1.0 / N
        real = np.zeros((N,)*3)
        imag = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        for i in range(0,N):
            real[i][mid][mid] = scale / (h ** 3)
        inputSpace = real + 1j * imag
    elif idi == GRIP_ID:
        print(' | Grip', end="")
        h = 1.0 / N
        r = np.int(N/4)
        real = np.zeros((N,)*3)
        imag = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        for i in range(0,N):
            for j in range(mid-r,mid+r):
                for k in range(mid-r, mid+r):
                    real[i][j][k] = scale / (h ** 3)
        inputSpace = real + 1j * imag
    return inputSpace