# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 02:03:59 2016

@author: Marko
"""
import numpy as np
from constants import *
from separated import jacobiPoissonSolve
from separated import gsPoissonSolve
from separated import sorPoissonSolve
from separated import multigridPoissonSolve

def poissonSolve(inputSpace, idm, N, precision):
    if idm == FFT_ID:
        return fftPoissonSolve(inputSpace, N)
    elif idm == JCB_ID:
        return iterativePoissonSolve(inputSpace, N, False, False, precision)
    elif idm == SEP_JCB_ID:
        return jacobiPoissonSolve(inputSpace, N, precision)
    elif idm == GS_ID:
        return iterativePoissonSolve(inputSpace, N, True, False, precision)
    elif idm == SEP_GS_ID:
        return gsPoissonSolve(inputSpace, N, precision)
    elif idm == SOR_ID:
        return iterativePoissonSolve(inputSpace, N, True, True, precision)
    elif idm == SEP_SOR_ID:
        return sorPoissonSolve(inputSpace, N, precision)
    elif idm == MTG_ID:
        return multigridPoissonSolve(inputSpace, N)
    else:
        return inputSpace

def multigridOldPoissonSolve(inputSpace, Np, precision): #Discards imaginary part of input
    print(' | Multigrid', end="")
    N = 2
    a = 1./N
    phi, phi2 = np.zeros([N+1,N+1,N+1],float), np.zeros([N+1,N+1,N+1],float)
    
    while (a >= 1./Np): #1024
        delt = 1
        first = True
        condition = True
        while (condition):
            phi2[:,:,0],phi2[:,:,N] = phi[:,:,0],phi[:,:,N]
            phi2[:,0,:],phi2[:,N,:] = phi[:,0,:],phi[:,N,:]
            phi2[0,:,:],phi2[N,:,:] = phi[0,:,:],phi[N,:,:]
        
            for i in range(1,N):
                for j in range(1,N):
                    for k in range(1,N):
                        phi2[i,j,k] = (phi[i+1,j,k] + phi[i-1,j,k] + phi[i,j+1,k] + phi[i,j-1,k] + phi[i,j,k+1] + phi[i,j,k-1] - a**2*inputSpace[i,j,k].real)/6         
                
                delt = np.amax(abs(phi-phi2))

            if first:
                first = False
                firstDiff = delt
            
            condition = delt > precision * firstDiff
            phi,phi2 = phi2,phi

        a /= 2; N *= 2
        phi2, phi3 = np.zeros([N+1,N+1,N+1],float), np.zeros([N+1,N+1,N+1],float)
        phi,phi3 = phi3,phi
    
    return phi3
    

def fftPoissonSolve(inputSpace, N):
    print(' | FFT', end="")
    outputSpace = np.fft.fftn(inputSpace)

    h = 1.0 / N    
    w = np.exp(2 * np.pi * 1j / N)
    wi = 1.0
    wj = 1.0
    wk = 1.0      

    processLater = []    
    
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                denom = wi + 1.0/wi + wj + 1.0/wj + wk + 1.0/wk - 6
                if not denom == 0:
                    outputSpace[i][j][k] = outputSpace[i][j][k] * h * h / denom
                else:
                    processLater.append([i,j,k]) 
                wk = wk * w
            wj = wj * w
        wi = wi * w                

    interpolate(processLater, outputSpace, N)    
    
    outputSpace = np.fft.ifftn(outputSpace)
    
    return outputSpace

def iterativePoissonSolve(inputSpace, N, gs, sor, precision):
    print(' |',(('SOR Gauss-Seidel' if sor else 'Gauss-Seidel') if gs else ('SOR Jacobi' if sor else 'Jacobi')), end="")
    w = 1
    if sor:
        w = 2.0 / (1 + np.pi / N)    
        print(' | w: ',w, end="")
    h = 1.0 / N
    real = np.zeros((N,)*3)
    imag = np.zeros((N,)*3)
    outputSpace = real + 1j * imag    
    converged = False
    prevOutputSpace = np.copy(outputSpace)
    cnt = 0
    limit = 1500
    maxdiff = 0
    firstdiff = 0
    while not converged:
        if cnt>=limit:
            print(' | Iter limit reached', end="")
            break
        for i in range (0,N):
            for j in range (0,N):
                for k in range (0,N):
                    curr = prevOutputSpace[i][j][k]
                    prevI = 0 if i==0   else (outputSpace if gs else prevOutputSpace)[i-1][j][k]
                    pastI = 0 if i==N-1 else prevOutputSpace[i+1][j][k]
                    prevJ = 0 if j==0   else (outputSpace if gs else prevOutputSpace)[i][j-1][k]
                    pastJ = 0 if j==N-1 else prevOutputSpace[i][j+1][k]
                    prevK = 0 if k==0   else (outputSpace if gs else prevOutputSpace)[i][j][k-1]
                    pastK = 0 if k==N-1 else prevOutputSpace[i][j][k+1]
                    fsor = ((1-w)*curr) if sor else 0
                    outputSpace[i][j][k] = fsor + w*(prevI + pastI + prevJ + pastJ + prevK + pastK - h * h * inputSpace[i][j][k] )/6
        diff = outputSpace - prevOutputSpace
        maxdiff = np.max(np.abs(diff))
        if cnt == 0:
            firstdiff = maxdiff
            print(' | First diff: ', firstdiff, end="")
        if maxdiff <= precision * firstdiff:
            converged = True
        temp = prevOutputSpace
        prevOutputSpace = outputSpace
        outputSpace = temp
        cnt += 1
    print(' | Last diff: ', maxdiff, end="")
    print(' | Iter num: ', cnt, end="")
    return outputSpace
    
def interpolate(processLater, outputSpace, N):
    for p in range(0,len(processLater)):
        elem = processLater[p]
        i = elem[0]
        j = elem[1]
        k = elem[2]
        
        cnt = 0
        csum = 0
        
        for n in range (np.max([0,i-1]),np.min([N-1,i+1])+1):
            for m in range (np.max([0,j-1]),np.min([N-1,j+1])+1):
                for l in range (np.max([0,k-1]),np.min([N-1,k+1])+1):
                    if not (i==n and j==m and k==l):
                        cnt += 1
                        csum += outputSpace[n][m][l]
                        
        outputSpace[i][j][k] = csum / cnt