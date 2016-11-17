import numpy as np
from outputgen import plotOutput

def initA(dim):
    No = dim ** 3
    # initialize the matrix
    A = np.zeros((No,)*2)
    for i in range(0,No):
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
            
    return A

def vcycle(A,f):
    # perform one v-cycle on the matrix A
    sizeF = np.size(A,axis=0);

    # size for direct inversion < 15
    #if sizeF < 15:
    if sizeF == 1:
        v = np.linalg.solve(A,f)
        #vpp = f / A
        #print(v)
        #print(vpp)
        
        #dim  = int(np.ceil(np.power(sizeF, 1/3)))
        #print(dim)
        #xp = v.reshape((dim, dim, dim)).transpose()
        #plotOutput(xp, dim, (int)(dim/2), "proba_mtg2"+np.str(iters))
        
        return v

    #b = - h**2 * f    
    
    # N1=number of Gauss-Seidel iterations before coarsening
    N1 = 5;
    v = np.zeros(sizeF);
    for numGS in range(N1):
        for k in range(sizeF):
            s1 = np.dot(A[k, :k], v[:k])
            s2 = np.dot(A[k, k + 1:], v[k + 1:])
            #v[k] = (b[k] - s1 - s2) / A[k, k]
            #v[k] = (f[k] - np.dot(A[k,0:k], v[0:k]) - np.dot(A[k,k+1:], v[k+1:]) ) / A[k,k];
            v[k] = (f[k] - s1 - s2) / A[k,k];
            
    # construct interpolation operator from next coarser to this mesh
    # next coarser has ((n-1)/2 + 1 ) points
    #print(sizeF)    
    #assert(sizeF%2 ==1)
    #sizeC = (int)((sizeF-1)/2+1)
    sizeC = (int)(sizeF/2)
    P = np.zeros((sizeF,sizeC));
    for k in range(sizeC):
        P[2*k,k] = 1; # copy these points
    for k in range(sizeC-1):
        P[2*k+1,k] = .5; # average these points
        P[2*k+1,k+1] = .5;
    
    # compute residual
    residual = f - np.dot(A,v)
    
    # project residual onto coarser mesh
    residC = np.dot(P.transpose(),residual)
    
    # Find coarser matrix (sizeC X sizeC)
    AC = np.dot(P.transpose(),np.dot(A,P))
    vC = vcycle(AC,residC);

    # extend to this mesh
    v = np.dot(P,vC)

    # N2=number of Gauss-Seidel iterations after coarsening
    N2 = 5;
    for numGS in range(N2):
        for k in range(sizeF):
            s1 = np.dot(A[k, :k], v[:k])
            s2 = np.dot(A[k, k + 1:], v[k + 1:])
            #v[k] = (b[k] - s1 - s2) / A[k, k]            
            #v[k] = (f[k] - np.dot(A[k,0:k], v[0:k]) - np.dot(A[k,k+1:], v[k+1:]) ) / A[k,k];
            v[k] = (f[k] - s1 - s2 ) / A[k,k];
        
    return v

#SOLVING

dim = 16
#N = 2**9+1
N = dim**3
x = np.linspace(0,1,N);
#h = x[1]-x[0] #SHIT
h = 1.0 / dim

# tridiagonal matrix
#A = np.diag(2.*np.ones(N)) - np.diag(np.ones(N-1), 1) - np.diag(np.ones(N-1), -1)
#A = A/h**2
A = initA(dim)
#print(A.shape)
#print (A)

#fg = np.ones(N, dtype=float) #rhs
fg = np.zeros(N)
mid = (dim * (int)(dim/2) + (int)(dim/2)) * dim + (int)(dim/2)
fg[mid] = -10 / h**3

print('F=',fg)

b = - h**2 * fg


print('B=',b)

#udirect = np.linalg.solve(A, b) # correct solution

u = np.zeros(N) # initial guess
for iters in range(100):
    print("Curr Iter: ", iters)
    r = b - np.dot(A,u)
    if np.linalg.norm(r)/np.linalg.norm(b) < 1.e-10:
        break
    du = vcycle(A, r)
    
    #xp = u.reshape((dim, dim, dim)).transpose()
    #plotOutput(xp, dim, (int)(dim/2), "proba_mtg2"+np.str(iters))
    
    u += du
    # INACE MOZDA KORISTAN PRINT    
    #print ("step %d, rel error=%e"% (iters+1, np.linalg.norm(u-udirect)/np.linalg.norm(udirect) ))

x = u
    
print('Iter: ', iters)
xp = x.reshape((dim, dim, dim)).transpose()

printSol = False
if printSol:
    print("Solution:")
    print(x)
    print("Solution (3D):")
    print(xp)

plotOutput(xp, dim, (int)(dim/2), "proba_mtg2")

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