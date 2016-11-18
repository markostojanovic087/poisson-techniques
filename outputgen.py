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
from algorithms import initA
import sys

def generateOutput(inputSpace, outputSpace, precision, N, ido, plnum, name):
    calculateError(inputSpace, outputSpace, N)    
    
    if ido == PLOT_ID:
        plotOutput(outputSpace, N, plnum, name)
    elif ido == CHCK_ID:
        checkOutput(outputSpace, N)

def calculateError(f, x, N):
    print(" | Checking", end="")
    sys.stdout.flush()
    
    h = 1.0 / N
    A = initA(N)    
    x = x.reshape((N**3))    
    f = f.reshape((N**3))    
    b = - h**2 * f
    
    error = A.dot(x) - b
    Signal = np.sum(x ** 2)
    Noise = np.sum(error ** 2)
    SNR = (10 * np.log10(Signal / Noise))  
    maxerr = np.max(np.abs(error))
    minerr = np.min(np.abs(error))
    avgerr = np.average(np.abs(error))

    print(" | SNR: ", SNR, (' Test Passed' if SNR>69.0 else ' Test Failed'), end="")    
    print(' | Max error: ', maxerr, end="")
    print(' | Min error: ', minerr, end="")
    print(' | Average error: ', avgerr, end="")
        
def plotOutput(outputSpace, N, plnum, name):
    print(' | Plotting')
    plnum = np.int(np.floor(plnum))
    plane = outputSpace[plnum]
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
    
    fig = plt.figure()
    plt.contour(outputSpace[plnum],50)
    fig.savefig("plots/contours/" + name + ".png",dpi=fig.dpi)
    
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