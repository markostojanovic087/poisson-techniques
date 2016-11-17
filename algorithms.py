# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 02:03:59 2016

@author: Marko
"""
import numpy as np
import sys
from constants import *
from separated import multigridPoissonSolve

def poissonSolve(inputSpace, idm, N, precision):
    if idm == FFT_ID:
        return fftPoissonSolve(inputSpace, N)
    elif idm == JCB_ID:
        return jacobiPoissonSolve(inputSpace, N, precision)
    elif idm == GS_ID:
        return gsPoissonSolve(inputSpace, N, precision)
    elif idm == SOR_ID:
        return sorPoissonSolve(inputSpace, N, precision)
    elif idm == MTG_ID:
        return multigridPoissonSolve(inputSpace, N)
    elif idm == PASS_ID:
        return passtrough(inputSpace)
    elif idm == MUL2_ID:
        return mul2(inputSpace)
    else:
        return inputSpace

def initA(dim):
    No = dim ** 3
    A = np.zeros((No,)*2)
    for i in range(0,No):
        ci = i  // dim**2
        cj = (i - ci * dim**2) // dim
        ck = i  %  dim
        curr = (ci * dim + cj) * dim +ck
        A[i][curr] = 6
        
        v = ck - 1
        if v>=0:       
            curr = (ci * dim + cj) * dim + v
            A[i][curr] = -1 
    
        v = ck + 1
        if v<dim:       
            curr = (ci * dim + cj) * dim + v
            A[i][curr] = -1 
            
        v = cj - 1
        if v>=0:       
            curr = (ci * dim + v) * dim + ck
            A[i][curr] = -1
        
        v = cj + 1
        if v<dim:
            curr = (ci * dim + v) * dim + ck
            A[i][curr] = -1
        
        v = ci - 1
        if v>=0:       
            curr = (v * dim + cj) * dim + ck
            A[i][curr] = -1
        
        v = ci + 1
        if v<dim:       
            curr = (v * dim + cj) * dim + ck
            A[i][curr] = -1
            
    return A

def calculateError(A, x, b):
    error = np.dot(A, x) - b
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
    #print('INPUT= ', inputSpace[][][])
    outputSpace = np.fft.fftn(inputSpace)
    #print('FFT= ', outputSpace)
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
                    #if outputSpace[i][j][k].real < 0:
                        #print('** [',i,'][',j,'][',k,']')
                else:
                    #print('[',i,'][',j,'][',k,']')
                    processLater.append([i,j,k]) 
                wk = wk * w
            wj = wj * w
        wi = wi * w                

    interpolate(processLater, outputSpace, N)    
    #print('POISS= ', outputSpace)
    outputSpace = np.fft.ifftn(outputSpace)
    #print('OUTPUT= ', outputSpace)
    return outputSpace
    
def jacobiPoissonSolve(f, dim, precision):
    print(' | Jacobi', end="")
    sys.stdout.flush()

    iter_limit = 1000
    N = dim ** 3
    h = 1.0/dim   
    f = f.reshape((N))    
    A = initA(dim)
    b = - h**2 * f
    
    x = np.zeros_like(b)
    for it_count in range(iter_limit):
        x_new = np.zeros_like(x)
    
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
    
        if np.allclose(x, x_new, atol=precision):
            break
    
        x = x_new
    
    print(' | Iterations: ', it_count, end="")
    
    calculateError(A, x, b)

    return x.reshape((dim, dim, dim)).transpose()    

def gsPoissonSolve(f, dim, precision):
    print(' | Gauss-Seidel', end="")
    sys.stdout.flush()

    iter_limit = 1000
    N = dim ** 3
    h = 1.0/dim   
    f = f.reshape((N))    
    A = initA(dim)
    b = - h**2 * f
    
    x = np.zeros_like(b)
    for it_count in range(iter_limit):
        x_new = np.zeros_like(x)
    
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
    
        if np.allclose(x, x_new, atol=precision):
            break
    
        x = x_new
    
    print(' | Iterations: ', it_count, end="")
    
    calculateError(A, x, b)

    return x.reshape((dim, dim, dim)).transpose() 

def sorPoissonSolve(f, dim, precision):
    print(' | SOR', end="")
    sys.stdout.flush()

    iter_limit = 1000
    N = dim ** 3
    h = 1.0/dim
    w = 2.0 / (1 + np.pi / dim)   
    f = f.reshape((N))    
    A = initA(dim)
    b = - h**2 * f
    
    x = np.zeros_like(b)
    for it_count in range(iter_limit):
        x_new = np.zeros_like(x)
    
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - w) * x[i] + w * (b[i] - s1 - s2) / A[i, i]
    
        if np.allclose(x, x_new, atol=precision):
            break
    
        x = x_new
    
    print(' | Iterations: ', it_count, end="")
    
    calculateError(A, x, b)

    return x.reshape((dim, dim, dim)).transpose() 
  
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

def passtrough(inputspace):
    return np.array(inputspace)
    
def mul2(inputspace):
    return inputspace*0.25