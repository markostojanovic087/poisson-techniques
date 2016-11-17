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

def generateOutput(outputSpace, precision, N, ido, plnum, name):
    if ido == PLOT_ID:
        plotOutput(outputSpace, N, plnum, name)
        
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