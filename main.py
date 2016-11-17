# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:36:37 2016

@author: Marko
"""
from inputgen import generateInput
from outputgen import generateOutput
from algorithms import poissonSolve
import numpy as np
from constants import *
import sys, getopt
import time

def main(argv):
    n = 1
    N = 16
    idi = POINT_ID
    idml = [MTG_ID] 
    ido = PLOT_ID
    scale = -10
    precision = 1e-10
    try:
        opts, args = getopt.getopt(argv,"hn:N:s:p:",["precision=", "ido=", "idi=", "idm=", "help", "scale=", "plnum="])
    except getopt.GetoptError:
        print('main.py BAD ARGS')
        sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
           print('main.py HELP')
           sys.exit()
       elif opt in ("--idi"):
           idi = arg
       elif opt in ("--ido"):
           ido = arg
       elif opt in ("--idm"):
           idml = arg.split(',')
       elif opt in ("-n"):
           n = np.int(arg)   
       elif opt in ("-p","--precision"):
           precision = np.float(arg) 
       elif opt in ("-N"):
           N = np.int(arg)
       elif opt in ("--plnum"):
           plnum = np.int(arg)
       elif opt in ("-s","--scale"):
           scale = np.int(arg)

    plnum = N/2       
    cumulativeTime = 0

    if ALL_ID in idml:
        idml = np.concatenate(([FFT_ID, JCB_ID, GS_ID, SOR_ID, MTG_ID],idml))
        idml = np.delete(idml, np.where(idml == ALL_ID))
        print(idml)
     
    for idm in idml:
        for i in range(0,n): 
            name = np.str(i+1) + '_' + np.str(N) + '_' + idi + '_' + idm + '_' + np.str(precision)
            print("{:9}".format(i+1), '/', n, ' | ',N,'x',N,'x',N, end="")
            inputSpace = generateInput(idi, N, scale)
            startTime = time.time()
            outputSpace = poissonSolve(inputSpace, idm, N, precision)
            endTime = time.time()
            executionTime = endTime - startTime
            print(" | ", "{:10.15f}".format(executionTime), end="")
            cumulativeTime = cumulativeTime + executionTime  
            generateOutput(outputSpace, precision, N, ido, plnum, name)

    print("Total execution time: %s seconds" % (cumulativeTime))
    averageTime = cumulativeTime / n
    print('Average time by input set: %s seconds' % averageTime)
if __name__ == "__main__":
    main(sys.argv[1:])