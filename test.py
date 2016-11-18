# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 04:24:15 2016

@author: Marko
"""
from scipy.sparse import lil_matrix
No = 32
A = lil_matrix((No, No))
print(A.shape)
A[0,0] = 1
print('A=',A[0,0])
#print('A=',A[0])
