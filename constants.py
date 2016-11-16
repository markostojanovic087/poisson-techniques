# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:31:57 2016

@author: Marko
"""

#Input generating method ids
RANDOM_ID   = "rnd"
POINT_ID    = "pt"
STICK_ID    = "st"
GRIP_ID     = "gr"

#Method ids for solving the equation
FFT_ID      = "fft"     #Fast Fourier transformation
JCB_ID      = "jcb"     #Jacobi
SEP_JCB_ID  = "sjcb"    #Jacobi
GS_ID       = "gs"      #Gauss-Seidel
SEP_GS_ID   = "sgs"     #Gauss-Seidel
SOR_ID      = "sor"     #Successive over relaxation
SEP_SOR_ID  = "ssor"    #Successive over relaxation
MTG_ID      = "mtg"     #Multigrid
PASS_ID     = "pass"
MUL2_ID     = "mul2"

#Output generating method ids
PLOT_ID     = "plot" #Plotting the image
CHCK_ID     = "chck" #Compares all succesive outputs - SNR
RELERR_ID   = "rerr" #Compares all succesive outputs - Relative error
ABSERRN_ID  = "aerr"

#Compute every combination
ALL_ID      = "all"