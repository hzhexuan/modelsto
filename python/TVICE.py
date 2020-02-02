import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def logerfc(t):
#
# out = logerfc(t): compute an accurate a estimate of log(erfc(t))
#

    out = np.zeros(t.shape) 
    id = t < 20.
    out[id] = np.log(scipy.special.erfc(t[id])) 
    
    c = np.cumprod(np.arange(1,16,2)/2) 
    t2n0 = t[id==0]**2
    t2n = np.copy(t2n0) 
    S = np.ones((t2n.size,)) 
    
    p = -1
    for n in range(8):
        S = S + (p * c[n]) / t2n 
        t2n = t2n*t2n0 
        p = -p 
    out[id==0] = -t2n0 + np.log(S/(t[id==0]*np.sqrt(np.pi))) 
    return out


def logerf2(a,b):
    #
    # usage: out = logerf2(a,b) with a < b
    # computes an accurate estimate of log(erf(b)-erf(a))

    a0 = np.copy(a)
    id = (b < 0)
    a[id] = -b[id]
    b[id] = -a0[id]

    out = np.zeros(a.shape) 
    id1 = (b-a)/(np.abs(a)+np.abs(b)) < 1e-14 
    out[id1] = np.log(2*(b[id1]-a[id1])/np.sqrt(np.pi)) - b[id1]**2 

    id2 = (id1==0) & (a<1) 
    out[id2] = np.log(scipy.special.erf(b[id2])-scipy.special.erf(a[id2])) 

    id3 = (id1==0) & (id2==0) 
    m = logerfc(b[id3]) 
    out[id3] = m + np.log(np.expm1(logerfc(a[id3])-m)) 
    
    return out


##################################    
##### WRITE THE FOLLOWING FUNCTION...
##################################
def padding(im):
    ny, nx = im.shape
    im_padded = np.zeros((ny+2, nx+2))
    im_padded[1:-1,1:-1] = im

    im_padded[0,1:-1] = im[0]
    im_padded[-1,1:-1] = im[-1]
    im_padded[1:-1,0] = im[:,0]
    im_padded[1:-1,-1] = im[:,-1]
    
    im_padded[0,0] = im[0,0]
    im_padded[0,-1] = im[0,-1]
    im_padded[-1,0] = im[-1,0]
    im_padded[-1,-1] = im[-1,-1]
    return im_padded

def calculate_logX(t, neibor, sig, lambd):
    a, b, c, d = neibor[0], neibor[1], neibor[2], neibor[3]
    X_minus2 = 2*lambd*(2*(t+2*lambd*sig**2)-a-b) * logerfc((t-a+4*lambd*sig**2)/sig/np.sqrt(2))
    X_minus1 = lambd*2*(t-b+lambd*sig**2) * logerf2((a-t-2*lambd*sig**2)/sig/np.sqrt(2), (b-t-2*lambd*sig**2)/sig/np.sqrt(2))
    X_0 = logerf2((b-t)/sig/np.sqrt(2), (c-t)/sig/np.sqrt(2))
    X_1 = lambd*2*(c-t+lambd*sig**2) * logerf2((c-t-2*lambd*sig**2)/sig/np.sqrt(2), (d-t-2*lambd*sig**2)/sig/np.sqrt(2))
    X_2 = 2*lambd*(c+d-2*(t-2*lambd*sig**2)) * logerfc((d-t+4*lambd*sig**2)/sig/np.sqrt(2))
    return X_minus2, X_minus1, X_0, X_1, X_2
def tvice(u0,sig,lambd,niter): 
# usage: out = tvice(u0,sigma,lambda,niter) 
# TV-ICE denoising algorithm (vectorized version)

    u = np.copy(u0)
    ny, nx = u.shape
    for k in range(niter):
        u = padding(u)
        #t = u[1:-1,1:-1]
        t = u0
        nebor = np.stack([u[1:-1,:-2],u[1:-1,2:],u[:-2,1:-1],u[2:,1:-1]], axis=0)
        nebor = nebor.reshape([-1,ny*nx])
        nebor.sort(axis=0)
        nebor = nebor.reshape([-1,ny,nx])
        X_minus2, X_minus1, X_0, X_1, X_2 = calculate_logX(t,nebor,sig,lambd)
        M = np.max(np.stack([X_minus2, X_minus1, X_0, X_1, X_2]),axis=0)
        X_minus2 = np.exp(X_minus2-M)
        X_minus1 = np.exp(X_minus1-M)
        X_0 = np.exp(X_0-M)
        X_1 = np.exp(X_1-M)
        X_2 = np.exp(X_2-M)
        u[1:-1,1:-1] = u0 + 2*lambd*sig**2 * (2*X_minus2+X_minus1-X_1-2*X_2)/(X_minus2+X_minus1+X_0+X_1+X_2)
        u = u[1:-1,1:-1]
        print(u[3,3])               
    return u
