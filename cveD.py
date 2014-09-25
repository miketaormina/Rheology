"""A module for analyzing diffusion data, calculating D and verifying free difusion behavior as in Vestergaard 2014.

Functions:
	getData -- loads data from a .mat file into a scipy object.
	cveD -- calculates the covariance-based estimator for the diffusion coefficient and localization variance. It also returns the variance in D, signal to noise ratio, and the computed power spectrum of displacement values for use with gammaChiSquare function.
	theoryP -- computes the theoretical power spectrum for a freely diffusing particle.
	boxAv -- computes a box averaged array, useful for comparing the power spectrum to it's theoretical value.
	gammaChiSquare -- compares an inpyt numpy array to a gamma distribution of shape 1/2 and scale 2 and returns a p value and reduced chi^2 value. Useful for verifying free difusion.
	D2eta -- computes a viscosity in Pa*s for a D in um^2/s, R in um, and T in Celcius

Dependencies:
	numpy, matplotlib, scipy

References:
	[1] Optimal estimation of diffusion coefficients from single-particle trajectories. Vestergaard, et al PRE (2014)
"""


import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import scipy.io
from scipy.fftpack import dst
import scipy.stats as ss
import scipy.constants as const


def getData(filename):
	"""Returns data from a .mat file"""
	#Filename is a string and indicates the path to a matlab file containing particle track info in objs_link (from RP tracking GUI)
	data = scipy.io.loadmat(filename, matlab_compatible=True)
	return data

def cveD(x, dt, scale):
	"""Computes the covariance-based estimator for D and localizationVariance, as well as variance in D, SNR, and Pxk
	
	Keyword arguments:
	x -- a 1Dnumpy array of particle track data in pixels
	dt -- the time step of data
	scale -- um per pixel for x

	Returns:
	[D, localizationVariance, varianceD, SNR], Pxk
	"""
	# For details, see [1]
	# Note that R is hard coded to 1/6, which should be the usual value for a shutter that is kept open during the entire time lapse [1]
	R = 1./6
	N = x.size
	dx = np.diff(scale*x) #(has N - 1 elements)

	#An array of elements (deltaX_n)*(deltaX_n-1) (should have N - 2 elements so just do the calculation and then truncate)
	dxn_dxn1 = dx*np.roll(dx,-1)
	dxn_dxn1 = dxn_dxn1[0:-1]

	# If the localization variance is known, D can be computed using eq [1].16, which also changes the calculation for the variance in D.
	# This function does not currently support that functionality.
    
	D = np.mean(dx**2)/2/dt + np.mean(dxn_dxn1)/dt
	localizationVariance = R*np.mean(dx**2) + (2*R - 1)*np.mean(dxn_dxn1)
	
	ep = localizationVariance/D/dt - 2*R
	varianceD = D**2*((6 + 4*ep + 2*ep**2)/N + 4*(1+ep)**2/N**2)
	SNR = np.sqrt(D*dt/localizationVariance)
    
	dxk = dt/2*dst(dx,1)
	Pxk = 2*dxk**2/(N + 1)/dt
    
	return np.array([D, localizationVariance, varianceD, SNR]), Pxk

def theoryP(D, dt, localizationVariance, N):
	"""Computes the theoretical power spectrum for a freely diffusing particle (eq 10 in [1])

	Keyword arguments:
	D -- diffusion constant
	dt -- time step
	localizationVariance -- 
	N -- total number of time steps to generate

	Returns:
	theoryPxk -- theoretical power spectrum of a freely diffusing particle
	"""
	# There might be a discrepency in N:
	# there are technically N - 1 values of k since it is indexing deltaX
	k = np.arange(0,N)
	#Hard code R as 1/6
	R = 1./6
	#Note that R is hard coded here as 1/6
	return 2*D*dt**2 + 2*(localizationVariance*dt - 2*D*R*dt**2)*(1 - np.cos(pi*k/(N+1)))

def boxAv(x, boxSize):
	"""Downsamples x with boxSize points per bin (last bin is padded with NaN)"""
	N = float(x.size)
	numBoxes = np.ceil(N/boxSize)
	lastBox = np.mod(N,boxSize)

	if lastBox != 0:
		pad = np.nan*np.zeros(boxSize - lastBox)
		xpadded = np.append(x,pad)
		binAv = np.nanmean(np.reshape(xpadded,(numboxes,boxSize)),1)
	else:
		binAv = np.nanmean(np.reshape(x,(numboxes,boxSize)),1)

	return binAv

def gammaChiSquared(n, bins):
	"""Calculates the chi^2 test statistic and p value against a gamma(1/2,2) distribution
	
	Keyword arguments:
	n -- pdf values of a histogram (as from pyplot.hist(x,N,normed=True)
	bins -- bins locations

	Returns:
	p -- p value
	X2 -- reduced chi^2 value
	"""
	# Useful for testing free diffusion
	# input n should be a pdf from plt.hist(... normed=True) such that sum(n*binwidth) = 1
	#
	grain = 1000.
	binwidth = np.diff(bins)[0]
    
	bins2 = binwidth/grain*(1+np.arange(0,grain*n.size))
    
	temp = scipy.stats.gamma.pdf(bins2,0.5, scale=2)
	temp = np.reshape(temp,(n.size,grain))*binwidth/grain
	expected = np.reshape(np.sum(temp,1),(n.size))
        
	observed = n*binwidth
	plt.bar(bins[0:-1],observed,binwidth)
	plt.plot(binwidth/2 + bins[0:-1],expected,'r', lw=5)
    
	p, X2 = scipy.stats.chisquare(observed, expected, 0)
	#temp = (observed - expected)**2/expected
	#X2 = np.sum(temp)
	return p, X2
    
    
def D2eta(D, R, T):
	"""Computes the viscosity of a fluid in cP

	Keyword arguments:
	D -- diffusion constant in um^2/s
	R -- tracked particle radius in um
	T -- temperature in C

	Returns:
	eta: viscosity in Pa*s (divide by 1000 for Pa*s)
	"""
	k = scipy.constants.k
	T = scipy.constants.C2K(T)
	R = R*scipy.constants.micro
	D = D*scipy.constants.micro**2
	return k*T/6/pi/R/D*1000
