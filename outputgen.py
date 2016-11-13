# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 02:54:23 2016

@author: Marko
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from constants import *
import sys

def generateOutput(outputSpace, N, ido, plnum, name):
    if ido == PLOT_ID:
        plotOutput(outputSpace, N, plnum, name)
    elif ido == CHCK_ID:
        checkOutput(outputSpace, N)
        
def plotOutput(outputSpace, N, plnum, name):
    print(' | Plotting')
    plane = outputSpace[np.int(np.floor(plnum))]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, N, 1)
    X, Y = np.meshgrid(x, y)
    Z = plane.real.reshape(X.shape)

    ax.plot_surface(X, Y, Z,             # data values (2D Arryas)
                    rstride=2,           # row step size
                    cstride=2,           # column step size
                    cmap=cm.hsv,        # colour map
                    linewidth=1,         # wireframe line width
                    antialiased=True)
    #ax.plot_wireframe(X, Y, Z, cmap=cm.jet)        
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('V')
    
    fig.savefig("plots/" + name + ".png",dpi=fig.dpi)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(prevOutputSpace=[], errorLog = open("output/errorLog.txt","w"))
def checkOutput(outputSpace, N):
    if len(checkOutput.prevOutputSpace) == 0:
        checkOutput.prevOutputSpace = outputSpace

    diff = np.abs(np.resize(outputSpace,checkOutput.prevOutputSpace.shape) - checkOutput.prevOutputSpace) 
    
    S = np.sum(checkOutput.prevOutputSpace.real ** 2 + checkOutput.prevOutputSpace.imag ** 2)
    Noise = np.sum(diff.real ** 2 + diff.imag ** 2)

    if not Noise == 0:
        SNR = (10 * np.log10(S / Noise))
    else:
        SNR = float("inf")
        
    print(" | SNR: ", SNR, (' Same' if SNR>69.0 else ' Different'))

    N = np.int(N if (N % 2) == 0 else N-1)

    checkOutput.errorLog.write('----------------------------------------------------------\n')
    
    if SNR<=69:
        for i in range(0,N):
            for j in range(0,N):
                for k in range(0,N):
                    checkOutput.errorLog.write('[' + np.str(i) + ']' + np.str(j) + '[' + np.str(k) + ']' + ' expected ' + np.str(checkOutput.prevOutputSpace[i][j][k]) + ' but found ' + np.str(outputSpace[i][j][k]) + '\n')
    else:
        checkOutput.errorLog.write("No error\n")
    
    checkOutput.prevOutputSpace = outputSpace