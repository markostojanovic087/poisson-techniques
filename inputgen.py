# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:22:42 2016

@author: Marko
"""

import numpy as np
from constants import *
from outputgen import plotOutput

def generateInput(idi, N, scale):  
    inputSpace = []
    if idi == RANDOM_ID:
        print(' | Random', end="")
        inputSpace = np.random.rand(N, N, N)
        inputSpace = 2 * inputSpace * scale - scale
    elif idi == POINT_ID:
        print(' | Point', end="")
        h = 1.0 / N
        inputSpace = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        inputSpace[mid][mid][mid] = scale / (h ** 3)
    elif idi == PTPER_ID:
        print(' | Periodic Points', end="")
        h = 1.0 / N
        inputSpace = np.zeros((N,)*3)
        pos = int(np.round(N/4))
        mid = int(np.round(N/2))
        inputSpace[mid][pos][pos] = scale / (h ** 3)
        inputSpace[mid][N-pos][N-pos] = - scale / (h ** 3)
        #plotOutput(inputSpace,N,round(N/2),"input")
    elif idi == STICK_ID:
        print(' | Stick', end="")
        h = 1.0 / N
        inputSpace = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        for i in range(0,N):
            inputSpace[i][mid][mid] = scale / (h ** 3)
    elif idi == STPER_ID:
        print(' | Periodic Stick', end="")
        h = 1.0 / N
        inputSpace = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        pos = int(np.round(N/4))
        for i in range(0,N):
            inputSpace[i][pos][pos] = scale / (h ** 3)
            inputSpace[i][N-pos][N-pos] = - scale / (h ** 3)
    elif idi == GRIP_ID:
        print(' | Grip', end="")
        h = 1.0 / N
        r = np.int(N/4)
        inputSpace = np.zeros((N,)*3)
        mid = np.int(np.floor(N/2))
        for i in range(0,N):
            for j in range(mid-r,mid+r):
                for k in range(mid-r, mid+r):
                    inputSpace[i][j][k] = scale / (h ** 3)
    elif idi == PERIODIC_ID:
        inputSpace = np.zeros((N,)*3)
        for i in range(0,N):
            for j in range(0,N):
                for k in range(0, N):
                    inputSpace[i][j][k] = np.sin(2*np.pi*(i+j+k)/(3*N))
    return inputSpace