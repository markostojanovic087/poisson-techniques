# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:56:27 2016

@author: Marko
"""
import numpy as np
from outputgen import plotOutput

def multi(outputS, inputS, i, g, h, Vd, Vu):
    N = np.int(np.power(2, g-i))     
    if i==g:
        outputS[0][0][0] = - h * h * inputS[0][0][0] / 6
    else:    
        for j in range(0,Vd):
            outputS = gsIteration(outputS, inputS, N, h)

        r = computeResidual(outputS, inputS, N, h)
        
        d = T(multi(complexZeros(np.int(N/2),3), R(r, N), i+1, g, h, Vd-2, Vu-2), np.int(N / 2))
        outputS = outputS + d
        
        for j in range(0, Vu):
            outputS = gsIteration(outputS, inputS, N, h)           
    
    return outputS

def computeResidual(outputS, inputS, N, h):
    r = complexZeros(N,3)
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
    rr = complexZeros(M,3)
    for i in range (0,M):
        for j in range (0,M):
            for k in range (0,M):
                ni = i * 2
                nj = j * 2
                nk = k * 2
                rr[i][j][k] = ( r[ni+1][nj][nk] + r[ni][nj+1][nk] + r[ni+1][nj+1][nk] + r[ni][nj][nk+1] + r[ni+1][nj][nk+1] + r[ni][nj+1][nk+1] + r[ni+1][nj+1][nk+1] ) / 8
                
    return rr
    
def T(outputS, N):
    p = complexZeros(N*2,3)
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                ni = i * 2
                nj = j * 2
                nk = k * 2
                p[ni][nj][nk] = p[ni+1][nj][nk] = p[ni][nj+1][nk] = p[ni+1][nj+1][nk] = p[ni][nj][nk+1] = p[ni+1][nj][nk+1] = p[ni][nj+1][nk+1] = p[ni+1][nj+1][nk+1] = outputS[i][j][k]
    
    return p
    
def multigridPoissonSolve(inputS, N):
    h = 1.0 / N
    Vd = 2 * np.int(np.log2(N))
    Vu = 2 * np.int(np.log2(N))
    print(' | sep-Multigrid', end="")
    outputS  = complexZeros(N,3)
    return multi(outputS, inputS, 0, np.int(np.log2(N)), h, Vd, Vu)

def gsMultiIteration(outputS, inputS, N, h):
    for i in range (1,N-1):
        for j in range (1,N-1):
            for k in range (1,N-1):
                outputS[i][j][k] = ( outputS[i-1][j][k] + outputS[i+1][j][k] + outputS[i][j-1][k] + outputS[i][j+1][k] + outputS[i][j][k-1] + outputS[i][j][k+1] - h * h * inputS[i][j][k] )/6
    
    return outputS

def jacobiPoissonSolve(inputS, N, precision):
    limit = 1500
    h = 1.0 / N    
    cnt = 0
    maxdiff = 0
    firstdiff = 0
    converged = False
    
    print(' | sep-Jacobi', end="")
    outputS  = complexZeros(N,3)    
    poutputS = complexZeros(N,3)    
    
    while not converged:
        if cnt>=limit:
            print(' | Iter limit reached', end="")
            break
        for i in range (0,N):
            for j in range (0,N):
                for k in range (0,N):
                    prevI = 0 if i==0   else poutputS[i-1][j][k]
                    pastI = 0 if i==N-1 else poutputS[i+1][j][k]
                    prevJ = 0 if j==0   else poutputS[i][j-1][k]
                    pastJ = 0 if j==N-1 else poutputS[i][j+1][k]
                    prevK = 0 if k==0   else poutputS[i][j][k-1]
                    pastK = 0 if k==N-1 else poutputS[i][j][k+1]
                    outputS[i][j][k] = (prevI + pastI + prevJ + pastJ + prevK + pastK - h * h * inputS[i][j][k] )/6
        
        diff = outputS - poutputS
        maxdiff = np.max(np.abs(diff))
        if cnt == 0:
            print(' | First diff: ', firstdiff, end="")
        if maxdiff <= precision:
            converged = True
        poutputS, outputS = outputS, poutputS
        cnt += 1
    print(' | Last diff: ', maxdiff, end="")
    print(' | Iter num: ', cnt, end="")
    return outputS
   
def gsPoissonSolve(inputS, N, precision):
    limit = 1500
    h = 1.0 / N    
    cnt = 0
    maxdiff = 0
    firstdiff = 0
    converged = False
    
    print(' | sep-Gauss-Seidel', end="")
    outputS  = complexZeros(N,3)
    poutputS = complexZeros(N,3)
    
    while not converged:
        if cnt>=limit:
            print(' | Iter limit reached', end="")
            break
        
        for i in range (0,N):
            for j in range (0,N):
                for k in range (0,N):
                    prevI = 0 if i==0   else outputS[i-1][j][k]
                    pastI = 0 if i==N-1 else outputS[i+1][j][k]
                    prevJ = 0 if j==0   else outputS[i][j-1][k]
                    pastJ = 0 if j==N-1 else outputS[i][j+1][k]
                    prevK = 0 if k==0   else outputS[i][j][k-1]
                    pastK = 0 if k==N-1 else outputS[i][j][k+1]
                    outputS[i][j][k] = (prevI + pastI + prevJ + pastJ + prevK + pastK - h * h * inputS[i][j][k] )/6
        
        diff = outputS - poutputS
        maxdiff = np.max(np.abs(diff))
        if cnt == 0:
            print(' | First diff: ', firstdiff, end="")
        if maxdiff <= precision:
            converged = True
        poutputS = np.copy(outputS)
        cnt += 1
    print(' | Last diff: ', maxdiff, end="")
    print(' | Iter num: ', cnt, end="")
    return outputS

def gsIteration(outputS, inputS, N, h):
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                prevI = 0 if i==0   else outputS[i-1][j][k]
                pastI = 0 if i==N-1 else outputS[i+1][j][k]
                prevJ = 0 if j==0   else outputS[i][j-1][k]
                pastJ = 0 if j==N-1 else outputS[i][j+1][k]
                prevK = 0 if k==0   else outputS[i][j][k-1]
                pastK = 0 if k==N-1 else outputS[i][j][k+1]
                outputS[i][j][k] = (prevI + pastI + prevJ + pastJ + prevK + pastK - h * h * inputS[i][j][k] )/6
    
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

def sorPoissonSolve(inputS, N, precision):
    limit = 1500
    h = 1.0 / N    
    cnt = 0
    maxdiff = 0
    firstdiff = 0
    converged = False
    w = 2.0 / (1 + np.pi / N)    
    print(' | w: ',w, end="")
        
    print(' | sep-SOR', end="")
    outputS  = complexZeros(N,3)
    poutputS = complexZeros(N,3)
    
    while not converged:
        if cnt>=limit:
            print(' | Iter limit reached', end="")
            break
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
        
        diff = outputS - poutputS
        maxdiff = np.max(np.abs(diff))
        if cnt == 0:
            print(' | First diff: ', firstdiff, end="")
        if maxdiff <= precision:
            converged = True
        poutputS = np.copy(outputS)
        cnt += 1
    print(' | Last diff: ', maxdiff, end="")
    print(' | Iter num: ', cnt, end="")
    return outputS    

def complexZeros(N,s):
    real = np.zeros((N,)*s)
    imag = np.zeros((N,)*s)
    return real + 1j * imag  