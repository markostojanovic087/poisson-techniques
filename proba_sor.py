# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 08:11:08 2016

@author: Marko
"""

from outputgen import plotOutput

import numpy as np

ITERATION_LIMIT = 1000

dim = 32
N = dim ** 3 
precision = 1e-3
w = 2.0 / (1 + np.pi / dim)    

print('w = ',w)
print('dim = ', dim)

# calculate h
h = 1.0/dim

# initialize f
f = np.zeros(N)
mid = (dim * (int)(dim/2) + (int)(dim/2)) * dim + (int)(dim/2)
f[mid] = -10 / h ** 3

# initialize the matrix
A = np.zeros((N,)*2)
for i in range(0,N):
    ci = i  // dim**2
    cj = (i - ci * dim**2) // dim
    ck = i  %  dim
    curr = (ci * dim + cj) * dim +ck
    #print('[',ci,',',cj,',',ck,'] = ', curr)
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

#print("A:")
#print(A)
    
#A = np.array([[6., -1., -1., 0., -1., 0., 0., 0],
 #             [-1., 6., 0., -1., 0., -1., 0., 0],
  #            [-1., 0., 6., -1., 0., 0., -1., 0],
   #           [0., -1., -1., 6., 0., 0., 0., -1.],
    #          [-1., 0., 0., 0., 6., -1., -1., 0],
     #         [0., -1., 0., 0., -1., 6., 0., -1.],
      #        [0., 0., -1., 0., -1., 0., 6., -1.],
       #       [0., 0., 0., -1., 0., -1., -1., 6]])
    
# initialize the RHS vector
b = - h**2 * f

# prints the system
printSys = False
if printSys:
    print("System:")
    for i in range(A.shape[0]):
        row = ["{}*x{}".format(A[i, j], j + 1) for j in range(A.shape[1])]
        print(" + ".join(row), "=", b[i])
    print()

x = np.zeros_like(b)
for it_count in range(ITERATION_LIMIT):
    #print("Current solution:", x)
    x_old = np.array(x)

    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (1 - w) * x[i] + w * (b[i] - s1 - s2) / A[i, i]

    if np.allclose(x_old, x, atol=precision):
        break

    #x = x_new

print('Iter: ', it_count)
xp = x.reshape((dim, dim, dim)).transpose()

printSol = False
if printSol:
    print("Solution:")
    print(x)
    print("Solution (3D):")
    print(xp)

plotOutput(xp, dim, (int)(dim/2), "proba_sor")

error = np.dot(A, x) - b

printErr = False
if printErr:
    print("Error:")
    print(error)
    
S = np.sum(x ** 2)
Noise = np.sum(error ** 2)
SNR = (10 * np.log10(S / Noise))  
print("SNR: ", SNR, (' Passed' if SNR>69.0 else ' Failed'))

maxerr = np.max(np.abs(error))
print('Max error: ', maxerr)
minerr = np.min(np.abs(error))
print('Min error: ', minerr)
avgerr = np.average(np.abs(error))
print('Average error: ', avgerr)