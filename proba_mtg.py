# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 03:47:49 2016

@author: Marko
"""

"""
THIS IS CORRECT MULTIGRID IMPLEMENTATION
"""

import numpy as np
from scipy.sparse import lil_matrix
import sys

def checkError(f, x, N):    
    h = 1.0 / N
    A = initA(N)    
    x = x.reshape((N**3))    
    f = f.reshape((N**3))    
    b = - h**2 * f
    
    error = A.dot(x) - b
    #Signal = np.sum(x ** 2)
    #Noise = np.sum(error ** 2)
    #SNR = (10 * np.log10(Signal / Noise))  
    #maxerr = np.max(np.abs(error))
    #minerr = np.min(np.abs(error))
    avgerr = np.average(np.abs(error))

    #print(" | SNR: ", SNR, (' Test Passed' if SNR>69.0 else ' Test Failed'), end="")    
    #print(' | Max error: ', maxerr, end="")
    #print(' | Min error: ', minerr, end="")
    #print(' | Average error: ', avgerr, end="")
    return avgerr

def initA(dim, dense=0):
    No = dim ** 3
    if dense == 0:
        A = lil_matrix((No,No))
    else:
        print('Dense')
        A = np.zeros((No,)*2)
    for i in range(0,No):
        ci = i  // dim**2
        cj = (i - ci * dim**2) // dim
        ck = i  %  dim
        curr = (ci * dim + cj) * dim +ck
        A[i,curr] = 6
        
        v = ck - 1
        if v>=0:       
            curr = (ci * dim + cj) * dim + v
            A[i,curr] = -1 
    
        v = ck + 1
        if v<dim:       
            curr = (ci * dim + cj) * dim + v
            A[i,curr] = -1 
            
        v = cj - 1
        if v>=0:       
            curr = (ci * dim + v) * dim + ck
            A[i,curr] = -1
        
        v = cj + 1
        if v<dim:
            curr = (ci * dim + v) * dim + ck
            A[i,curr] = -1
        
        v = ci - 1
        if v>=0:       
            curr = (v * dim + cj) * dim + ck
            A[i,curr] = -1
        
        v = ci + 1
        if v<dim:       
            curr = (v * dim + cj) * dim + ck
            A[i,curr] = -1
            
    return A

def fv(V):
    return V*2

def multi(outputS, inputS, i, g, h, Vd, Vu, w):
    N = np.int(np.power(2, g-i))     
    if i==g:
        outputS[0][0][0] = - h * h * inputS[0][0][0] / 6
    else:    
        for j in range(0,Vd):
            outputS = gsIteration(outputS, inputS, N, h, w)

        r = computeResidual(outputS, inputS, N, h)
        
        d = T(multi(np.zeros((np.int(N/2),)*3), R(r, N), i+1, g, h, fv(Vd), fv(Vu), w), np.int(N / 2))
        outputS = outputS + d
        
        for j in range(0, Vu):
            outputS = gsIteration(outputS, inputS, N, h, w)           
    
    return outputS

def computeResidual(outputS, inputS, N, h):
    r = np.zeros((N,)*3)
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                prevI = 0 if i==0   else outputS[i-1][j][k]
                pastI = 0 if i==N-1 else outputS[i+1][j][k]
                prevJ = 0 if j==0   else outputS[i][j-1][k]
                pastJ = 0 if j==N-1 else outputS[i][j+1][k]
                prevK = 0 if k==0   else outputS[i][j][k-1]
                pastK = 0 if k==N-1 else outputS[i][j][k+1]
                r[i][j][k] = -1 * ( prevI + pastI + prevJ + pastJ + prevK + pastK - 6 * outputS[i][j][k] ) / ( h * h ) + inputS[i][j][k]
    
    return r

def R(r, N):
    M = np.int(N / 2)
    rr = np.zeros((M,)*3)
    for i in range (0,M):
        for j in range (0,M):
            for k in range (0,M):
                ni = i * 2
                nj = j * 2
                nk = k * 2
                rr[i][j][k] = ( r[ni+1][nj][nk] + r[ni][nj+1][nk] + r[ni+1][nj+1][nk] + r[ni][nj][nk+1] + r[ni+1][nj][nk+1] + r[ni][nj+1][nk+1] + r[ni+1][nj+1][nk+1] ) / 8
                
    return rr
    
def T(outputS, N):
    p = np.zeros((N*2,)*3)
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                ni = i * 2
                nj = j * 2
                nk = k * 2
                p[ni][nj][nk] = p[ni+1][nj][nk] = p[ni][nj+1][nk] = p[ni+1][nj+1][nk] = p[ni][nj][nk+1] = p[ni+1][nj][nk+1] = p[ni][nj+1][nk+1] = p[ni+1][nj+1][nk+1] = outputS[i][j][k]
    
    return p

def multigridPoissonSolveA(inputS, N, precision):
    h = 1.0 / N
    Vd = np.int(np.log2(N))
    Vu = np.int(np.log2(N))
    w = 2.0 / (1 + np.pi / N)
    #Vd = 5
    #Vu = 5    
    print(' | Multigrid', end="")

    outputS  = np.zeros((N,)*3)
    
    iter_limit = int(np.ceil(precision))
    print(' | Iterations: ', iter_limit, end="")
    sys.stdout.flush()
    
    for it_count in range(iter_limit):
        outputS = multi(outputS, inputS, 0, np.int(np.log2(N)), h, Vd, Vu, w)
        #if checkError(inputS, outputS, N) < precision:
            #break
    
    return outputS

def gsMultiIteration(outputS, inputS, N, h):
    for i in range (1,N-1):
        for j in range (1,N-1):
            for k in range (1,N-1):
                outputS[i][j][k] = ( outputS[i-1][j][k] + outputS[i+1][j][k] + outputS[i][j-1][k] + outputS[i][j+1][k] + outputS[i][j][k-1] + outputS[i][j][k+1] - h * h * inputS[i][j][k] )/6
    
    return outputS
   


def gsIteration(outputS, inputS, N, h, w):
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                curr = outputS[i][j][k]
                prevI = 0 if i==0   else outputS[i-1][j][k]
                pastI = 0 if i==N-1 else outputS[i+1][j][k]
                prevJ = 0 if j==0   else outputS[i][j-1][k]
                pastJ = 0 if j==N-1 else outputS[i][j+1][k]
                prevK = 0 if k==0   else outputS[i][j][k-1]
                pastK = 0 if k==N-1 else outputS[i][j][k+1]
                fsor = ((1-w)*curr)
                outputS[i][j][k] = fsor + w*(prevI + pastI + prevJ + pastJ + prevK + pastK - h * h * inputS[i][j][k] )/6
    
    return outputS    
    
def opGsIteration(outputS, inputS, N, h):
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                prevI = 0 if i==0   else outputS[i-1][j][k]
                pastI = 0 if i==N-1 else outputS[i+1][j][k]
                prevJ = 0 if j==0   else outputS[i][j-1][k]
                pastJ = 0 if j==N-1 else outputS[i][j+1][k]
                prevK = 0 if k==0   else outputS[i][j][k-1]
                pastK = 0 if k==N-1 else outputS[i][j][k+1]
                outputS[i][j][k] = (prevI + pastI + prevJ + pastJ + prevK + pastK + h * h * inputS[i][j][k] )/6
    
    return outputS
