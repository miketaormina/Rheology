import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy.fft as fft

pi = np.pi

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



def newtonianfluid(x, a, tau):
    """Functional form for the decay of amplitude with frequency of a angular driven onject in a Newtonian fluid.
    """
    return a*np.sqrt(1/(1+(2*pi*x*tau)**2))

def newtonianfluidphase(f, tau):
    """Functional form of the phase difference between a driven oscilator and the driving field.

    Inputs:
	- x, frequencies over which to return deltaPhi
	- tau, characteristic time tau = 1/omega*
    Outputs:
    	- deltaPhi, phase lag for a Newtonian fluid, for a given tau
    """

    return np.arctan(-(2*pi*f*tau))


def fit_to_newtonian(F, A, p0=[20, 0.01]):
    """Fit amplitude of angular oscillation as a function of driving frequency to the theoretical behavior in a Newtonian Fluid. Non linear least squares is used through scipy.optimize.curvefit
    """
    try:
	    popt, pcov = curve_fit(newtonianfluid, F, A, p0)
    except RuntimeError:
	    popt = np.array([np.nan, np.nan])
	    pcov = np.nan
    return popt, pcov

def fit_phase_to_newtonian(F, P, p0=[0.01]):
    """Fit the phase lag of angular oscillator as a function of the driving frequency to the theoretical behavior in a Newtonian Fluid. Non linear least squres is used through scipy.optimize.curvefit
    """
    try:
	popt, pcov = curve_fit(newtonianfluidphase, F, P, p0)
    except RuntimeError:
	popt = np.array([np.nan])
	pcov = np.nan
    return popt, pcov


def correctedNewtonianFluid(x, a, tau):
    """Obsolete - creates the amplitude decay of a Newtonian Fluid with corrections for an oscillator that deviates from the small angle approximation.
    """
    i = np.complex(0,1)
    t = (x/x[0])**(1/(np.log(x[-1]/x[0])/np.log(30.)))
    #b = np.tan(a*pi/180)
    omega = 2*pi*x
    b = np.tan(a*pi/180)*np.exp(i*omega*t)
    bb = b**2
    #return a*np.sqrt(((1 + 0.5*bb*np.cos(2*omega*t))**2 +
    #                  (0.5*bb*np.sin(2*omega*t))**2)/((1 + 0.5*np.cos(2*omega*t))**2 +
    #                                               (omega*tau +
    #                                                0.5*bb*np.sin(2*omega*t))**2))
    return a*np.sqrt(((1 + 0.5*bb.real)**2 + (0.5*bb.imag)**2)/((1 + 0.5*bb.real)**2 + (tau*omega + 0.5*bb.imag)**2))

def fit_to_corrected_newtonian(F, A, p0=[20, 0.01]):
    """Obsolete - Fits to amplitude decay of a Newtonian Fluid with corrections for an oscillator that deviates from the small angle approximation.
    """
    popt, pcov = curve_fit(correctedNewtonianFluid, F, A, p0)
    return popt, pcov

def maxwellfluid(x, a, tau, c):
    return a*np.sqrt((1+(2*pi*x*tau)**2)/(1+(2*pi*x*tau)**2*(1+c)**2))

def maxwellfluidphase(x, tau, c):
	return (np.arctan(2*np.pi*x*tau) - np.arctan((1+c)*2*np.pi*x*tau))

def fit_to_maxwell(F, A, p0=[20, 0.001, 10]):
    """Fit amplitude of angular oscillation as a function of driving frequency to the theoretical behavior in a Maxwell Fluid. Non linear least squares is used through scipy.optimize.curvefit.
    """
    popt, pcov = curve_fit(maxwellfluid, F, A , p0)
    return popt, pcov


def frameInfo(data):
    
    frameNumber = data[:,0]
    lastFrame = frameNumber[-1].astype(int)
    droppedFrames = frameNumber[diff(frameNumber)>1].astype(int)
    dupedFrames = frameNumber[diff(frameNumber) <1].astype(int)
    
    thetaRaw  = data[:,7]
    secondsRaw = 60*data[:,5] + data[:,6]
    secondsRaw = secondsRaw - secondsRaw[0]
    
    return frameNumber,droppedFrames, dupedFrames, thetaRaw - mean(thetaRaw), secondsRaw


def reject_outliers(data, m=2):
    """Function that rejects values more than a given amount from the mean.
    Inputs:
        - data, numpy array of data to be inspected.
	- m[=2], multiple of the standard deviation above which data will be rejected.
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def construct_input(camSignal, magSignal, scopeTime, time, fInitial, fFinal,
                    sweepTime, holdTime, quadrant=0):
    """Produces a remapped version of the input signal such that it is sampled at the same points as the measured response (from image frames).

    Input:
        - camSignal, 1D array of the camera's logical signal for commencement of acquisition. This comes from the signal captured on the oscilloscope.
        - magSignal, 1D array of the voltage applied to the electromagnet. This is measured on the oscilloscope.
        - scopeTime, 1D array of the time data from the oscilloscope measurements.
        - time, 1D array of the measurement times (e.g. from the camera's timestamp).
        - fInitial, starting frequency of the chirp.
        - fFinal, ending frequency of the chirp.
        - sweepTime, time in seconds over which the frequency is chirped.
        - holdTime, time in seconds that the input signal remains at the final frequency at the end of the chirp.
        - quadrant[=0], The function attempts to find the quadrant, or phase, of the input signal but sometimes gets it wrong. If this is required (incase phase information is being calculated), the actual quadrant can be deduced and input manually here.
	    """
    fInitial = np.float(fInitial)
    fFinal = np.float(fFinal)
    sweepTime = np.float(sweepTime)
    holdTime = np.float(holdTime)
    camSignal = np.array(camSignal)
    magSignal = np.array(magSignal)
    scopeTime = np.array(scopeTime)

    if fInitial == fFinal:
        case = 'constant'
    else:
        case = 'sweep'

    if case == 'constant':
        _= scopeTime[camSignal>1]
        scopeTime = scopeTime - _[0]

        f = fInitial
        phi = 2*pi*f*time
    elif case == 'sweep':
        _ = scopeTime[camSignal>1]
        scopeTime = scopeTime - _[0]
    
        k = (fFinal/fInitial)**(1/sweepTime)
        f = fInitial*k**time
        f[time>=sweepTime] = fFinal
    
        phi = 2*pi*fInitial*(k**time - 1)/np.log(k)
        _ = phi[time<sweepTime]
        phi[time>=sweepTime] = _[-1] + 2*pi*fFinal*(time[time>=sweepTime]-max(time[time<sweepTime]))


    """
        Need to address the degeneracy in arcCos. try looking several (~10)
        points in advance and asking whether that value is higher or lower than
        at time 0.        
    """
    phi0 = 0
    _ = magSignal[camSignal>=1]/max(magSignal) 
    if quadrant == 0:
        __ = magSignal[camSignal<1]/max(magSignal)
        test1 = np.sign(_[0])
        test2 = np.concatenate((__[-10:0:1],_[0:10:1]),0)
        slopetest = np.sign(np.mean(np.diff(test2)))

        if np.logical_and(test1==1, slopetest==-1):
            quadrant = 1
        elif np.logical_and(test1==-1, slopetest==-1):
            quadrant = 2
        elif np.logical_and(test1==-1, slopetest==1):
            quadrant = 3
        elif np.logical_and(test1==1, slopetest==1):
            quadrant = 4
        else:
            print 'There seems to be a problem determining your quadrant, you should specify one in the function call'
            return


    


#    test2 = _[20] - _[0]
    
#    _ = magSignal[camSignal<1]/max(magSignal)
#    test3 = _[-1] - _[-20]

#    if np.sign(test2) == np.sign(test3):
#        #slopes agree, use test2
#        slopetest = test2
#    else:
#        #slopes disagree, use whichever has the larger magnitude
#        _ = np.array([test2, test3])
#        __ = max(np.absolute(_))
#        slopetest = _[np.absolute(_)==__]
#        if slopetest.size != 1:
#            print 'Problem with slopetest size (= ' + str(slopetest.size) + ')'
#            return



    if quadrant ==1:
        """You are in the first quadrant, use arccos as normal"""
        print 'I think you are in the 1st quadrant'
        phi0 = np.arccos(_[0])
    elif quadrant ==2:
        """You are in the second quadrant, use arccos as normal"""
        print 'I think you are in the 2nd quadrant'
        phi0 = np.arccos(_[0])
    elif quadrant ==3:
        """You are in the third quadrant, use arccos(-y) + pi"""
        print 'I think you are in the 3rd quadrant'
        phi0 = np.arccos(-_[0]) + pi
    elif quadrant ==4:
        """You are in the fourth quadrant, use arccos(-y) + pi"""
        print 'I think you are in the 4th quadrant'
        phi0 = np.arccos(-_[0]) + pi
    
    
    
   # phi0 = np.arccos(_[0]/max(magSignal))
    inputSignal = -np.cos(phi + phi0)
    
    plt.figure(figsize=(15,4))
    plt.plot(scopeTime, magSignal/max(magSignal), 'b')
    plt.plot(scopeTime, camSignal, 'r')
    plt.plot(time, -inputSignal,'k')
    #ylim((-1.1,1.1))
    plt.show()
    
    return phi0, phi, inputSignal, f


def interpolate_and_cull(beadPosition, referenceFrame, theta, seconds, frameNumber):
    """ Helper function to patch time points where the camera dropped a frame and identify the correct ellipse when
        more than one is found.
        
        Input:
        - beadPosition (data[:,10:]), the position of found ellipses in each frame, including false positives
        - referenceFrame, a frame number where the only ellipse found is the one of interest
        - theta (data[:,7]), array of tracked angles for each found ellipse. 
        - seconds, array of time when each ellipse is found.
        - frameNumber (data[:,0]), the number of each frame with identified ellipse.
        
        Outputs:
        - newtheta, original data with holes patched and false values removed
        - newseconds, correspondingly fixed time data
        - pseudoT, regularly spaced time points to match average frame rate. Useful for fft analysis
        
        """
    
    refPosition = beadPosition[np.int(referenceFrame)]
    _ = beadPosition - refPosition
    _ = _**2
    distanceFromRef = np.sqrt(_[:,0] + _[:,1])
    
    newtheta = np.zeros(np.int(frameNumber[-1]))
    newseconds = np.zeros(np.int(frameNumber[-1]))
    for n in range(1,np.int(frameNumber[-1])):
        _ = np.where(frameNumber == n)[0]
        if _.size == 0:
            newtheta[n-1] = np.nan
            newseconds[n-1] = np.nan
        else:
            temp = distanceFromRef[_]
            M = np.argmin(temp)
            _ = np.where(distanceFromRef == temp[M])[0]
            #print 'attempting to use ' + str(_) + ' as an index'
            newtheta[n-1] = theta[_[0]]
            newseconds[n-1] = seconds[_[0]]


    nans, x= nan_helper(newtheta)
    newtheta[nans]= np.interp(x(nans), x(~nans), newtheta[~nans])

    nans, x= nan_helper(newseconds)
    newseconds[nans]= np.interp(x(nans), x(~nans), newseconds[~nans])

    dt = np.mean(reject_outliers(np.diff(newseconds),2))
    _ = dt.size
    if dt.size !=1:
        print 'problem with dt: it is ' + str(_) + ' in length'
        return

    pseudoT = np.arange(0,int(frameNumber[-1])*dt,dt)
    newtheta = newtheta - np.nanmean(newtheta)

    _ = pseudoT.size
    __ = newtheta.size

    if _ > __:
        pseudoT = pseudoT[0:-1:1]
        print 'Warning: truncated one time point due to element mismatch'
    elif __ > _:
        newtheta = newtheta[0:-1:1]
        print 'Warning: truncated one angle point due to element mismatch'

    _ = pseudoT.size
    __ = newtheta.size


    if _ != __:
        print 'mismatch in time/theta length: ' + str(_) + ' vs ' + str(__)
        print 'total number of frames is ' + str(frameNumber[-1])
        return
    
    plt.figure(figsize=(15,4))
    plt.plot(pseudoT,newtheta,'k-')
    plt.plot(pseudoT[nans],newtheta[nans],'rs')
    #xlim((23.5,23.6))
    plt.show()

    return newtheta, newseconds, pseudoT, dt


def smooth_result(data, F, sigma):
    sigma = np.float(sigma)
    
    gauss = 1/(np.sqrt(2*pi)*sigma)*np.exp(-((F/sigma)**2)/2)
    gaussSum = sum(gauss)
    smooth = np.convolve(data,gauss/gaussSum,mode='same')
    
    return smooth

