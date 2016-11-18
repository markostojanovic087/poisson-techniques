# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:31:57 2016

@author: Marko
"""

#Input generating method ids
RANDOM_ID   = "rnd"
POINT_ID    = "pt"
PTPER_ID    = "ptper"
STICK_ID    = "st"
STPER_ID    = "stper"
GRIP_ID     = "gr"
PERIODIC_ID     = "per"

#Method ids for solving the equation
FFT_ID      = "fft"     #Fast Fourier transformation
JCB_ID      = "jcb"     #Jacobi
GS_ID       = "gs"      #Gauss-Seidel
SOR_ID      = "sor"     #Successive over relaxation
MTG_ID      = "mtg"     #Multigrid

#Output generating method ids
PLOT_ID     = "plot" #Plotting the image
CHCK_ID     = "chck" #Compares all succesive outputs - SNR

#Compute every combination
ALL_ID      = "all"