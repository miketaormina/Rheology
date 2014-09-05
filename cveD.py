import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import scipy.io
from scipy.fftpack import dst
import scipy.stats as ss
import scipy.constants as const

#%matplotlib tk

def getData(filename):
	#Filename is a string and indicates the path to a matlab file containing particle track info in objs_link (from RP tracking GUI)
	data = scipy.io.loadmat(filename, matlab_compatible=True)
	return data

def cveD(x, dt, scale):
	#Computes the Covariance based estimator of D, localization sigma, variance of D, Signal to noise, and the power spectrum of dx
	# For details, see Vestergaard (2014)
	# Note that R is hard coded to 1/6, which should be the usual value
	R = 1./6
	N = x.size
	dx = np.diff(x) #(has N - 1 elements)

	#An array of elements (deltaX_n)*(deltaX_n-1) (should have N - 2 elements so just do the calculation and then truncate)
	dxn_dxn1 = dx*np.roll(dx,-1)
	dxn_dxn1 = dxn_dxn1[0:-1]
    
	D = np.mean(dx**2)/2/dt + np.mean(dxn_dxn1)/dt
	localizationVariance = R*np.mean(dx**2) + (2*R - 1)*np.mean(dxn_dxn1)
	
	ep = localizationVariance/D/dt - 2*R
	varianceD = D**2*((6 + 4*ep + 2*ep**2)/N + 4*(1+ep)**2/N**2)
	SNR = np.sqrt(D*dt/localizationVariance)
    
	dxk = dt/2*dst(dx,1)
	Pxk = 2*dxk**2/(N + 1)/dt
    
	return np.array([D, localizationVariance, varianceD, SNR]), Pxk

def theoryP(D, dt, localizationVariance, N):
    #For a given D, dt, and sigma**2, returns equation (10) from Vestergaard with N points. There might be a discrepency in N:
    # there are technically N - 1 values of k since it is indexing deltaX
    k = np.arange(0,N)
    #Hard code R as 1/6
    R = 1./6
    #Note that R is hard coded here as 1/6
    return 2*D*dt**2 + 2*(localizationVariance*dt - 2*D*R*dt**2)*(1 - np.cos(pi*k/(N+1)))

def boxAv(x, boxSize):
    N = x.size
    numBoxes = np.ceil(N/boxSize)
    lastBox = np.mod(N,boxSize)
    binAv = np.zeros((numBoxes))
    for i in np.arange(0,numBoxes - 1):
        binAv[i] = np.mean(x[i*boxSize:(i+1)*boxSize - 1])
    
    if lastBox ==1:
        binAv[-1] = x[-1]
    else:
        binAv[-1] = np.mean(x[-1:-(lastBox):-1])
    return binAv

def gammaChiSquared(n, bins):
    # Calculate the X**2 test statistic of calculated Pxk histogram values to theory Pxk = 2D(dt**2) + 2(sigma**2dt - 2DR(dt**2)(1-cos(pik/(N+1))))
    # input n should be a pdf from plt.hist(... normed=True) such that sum(n*binwidth) = 1
    # 
    binwidth = np.diff(bins)[0]
    
    bins2 = binwidth/100.*(1+np.arange(0,100*n.size))
    
    temp = scipy.stats.gamma.pdf(bins2,0.5, scale=2)
    temp = np.reshape(temp,(n.size,100))*binwidth/100
    expected = np.reshape(np.sum(temp,1),(n.size))
        
    observed = n*binwidth
    plt.bar(bins[0:-1],observed,binwidth)
    plt.plot(binwidth/2 + bins[0:-1],expected,'r', lw=5)
    
    p, X2 = scipy.stats.chisquare(observed, expected, 0)
    #temp = (observed - expected)**2/expected
    #X2 = np.sum(temp)
    return p, X2
    
    
def D2eta(D, R, T):
    #Takes D in um**2/s, R in um, T in C, and returns eta in centipoise
    k = scipy.constants.k
    T = scipy.constants.C2K(T)
    R = R*scipy.constants.micro
    D = D*scipy.constants.micro**2
    return k*T/6/pi/R/D*1000
