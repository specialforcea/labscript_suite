"""
Created on Wed Sep 11 18:47:08 2013
@author: ispielman
QgasFunctions: mathematical functions used in quantum gas experiments. For example for fitting to data

Modified on Mon Dec 15 10:57:02 2014
@author: aputra
Correction made:
1.  TF function return becomes "p[0] + p[1]*condition**(3.0/2.0)". Previously p[1] is included in the
    power of (3/2), where p[1] is the TF peak.
2.  Added a whole bunch of more descriptive comments.
3.  Added puck2DSG and puckThermal2DSG which give 3 ellipse masks and thermal distribution masked with 3 ellipses
"""


import numpy as np
import scipy
import scipy.optimize
from scipy.interpolate import interp1d


# Functions for 1D and 2D fitting using IanStyle fit functions:
# 'xyVals' is a list of the independent variable arrays, 
# 'p' the parameter vector

def gaussian(xyVals,p) :
    # A gaussian peak with:
    #   Constant Background             : p[0]
    #   Peak height above background    : p[1]
    #   Central position                : p[2]
    #   Standard deviation              : p[3]
    return p[0]+p[1]*np.exp(-1*(xyVals[0]-p[2])**2/(2*p[3]**2))

def TF_only(xyVals,p) :
    # A Thomas Fermi peak with:
    #   Constant Background             : p[0]
    #   Peak height above background    : p[1]
    #   Central position                : p[2]
    #   Radius                          : p[3]
    condition = (1 - ((xyVals[0]-p[2])/p[3])**2)
    condition[condition < 0.0] = 0
    return p[0] + p[1]*(condition**(3.0/2.0))

def lorentzian(xyVals,p) :
    # A lorentzian peak with:
    #   Constant Background             : p[0]
    #   Peak height above background    : p[1]
    #   Central position                : p[2]
    #   Full Width at Half Maximum      : p[3]
    return p[0]+(p[1]/np.pi)/(1.0+((xyVals[0]-p[2])/p[3])**2)

def line(xyVals,p) :
    # A linear fit with:
    #   y-Intercept     : p[0]
    #   Slope           : p[1]
    return p[0]+p[1]*xyVals[0]

def power(xyVals,p) :
    # A power law fit with:
    #   Normalization   : p[0]
    #   Position offset : p[1]
    #   Power           : p[2]
    #   Constant        : p[3]
    return p[0]*(xyVals[0]-p[1])**p[2]+p[3]

def gaussian2D(xyVals,p) :
    # A 2D gaussian peak with:
    #   Constant Background             : p[0]
    #   Peak height above background    : p[1]
    #   X Central position              : p[2]
    #   X Standard deviation            : p[3]
    #   Y Central position              : p[4]
    #   Y Standard deviation            : p[5]
    return p[0] + p[1]*np.exp(-1*(xyVals[0]-p[2])**2/(2*p[3]**2)-1*(xyVals[1]-p[4])**2/(2*p[5]**2))

def TF_only2D(xyVals,p) :
    # A 2D gaussian peak with:
    #   Constant Background             : p[0]
    #   Peak height above background    : p[1]
    #   X Central position              : p[2]
    #   X width                         : p[3]
    #   Y Central position              : p[4]
    #   Y width                         : p[5]
    condition = (1.0 - ((xyVals[0]-p[2])/p[3])**2.0 - ((xyVals[1]-p[4])/p[5])**2.0)
    condition[condition < 0.0] = 0.0
    return p[0] + p[1]*(condition**(3.0/2.0))

def puck(xyVals, p) :       # an ellipse mask
    # returns 1.0 within a specified boundary. 0.0 everywhere else:
    #   X Central position  : p[0]
    #   X width / 2         : p[1]
    #   Y Central position  : p[2]
    #   Y width / 2         : p[3]
    condition = (1.0 - ((xyVals[0]-p[0])/p[1])**2.0 - ((xyVals[1]-p[2])/p[3])**2.0)
    condition[condition < 0.0] = 0.0
    condition[condition > 0.0] = 1.0
    return condition


def puck2DSG(xyVals,p) :    # three ellipse masks with common major and minor axes
    # returns 1.0 within a specified boundary. 0.0 everywhere else:
    #   MinOne (X,Y) Central position   : p[0][0,1]
    #   Zero (X,Y) Central position     : p[1][0,1]
    #   PlusOne (X,Y) Central position  : p[2][0,1]
    #   (X,Y) width / 2                 : p[3][0,1]
    xw = p[3][0]
    yh = p[3][1]
    m1 = (1.0 - ((xyVals[0]-p[0][0])/xw)**2.0 - ((xyVals[1]-p[0][1])/yh)**2.0)
    m1[m1 < 0.0] = 0.0
    z0 = (1.0 - ((xyVals[0]-p[1][0])/xw)**2.0 - ((xyVals[1]-p[1][1])/yh)**2.0)
    z0[z0 < 0.0] = 0.0
    p1 = (1.0 - ((xyVals[0]-p[2][0])/xw)**2.0 - ((xyVals[1]-p[2][1])/yh)**2.0)
    p1[p1 < 0.0] = 0.0
    condition = m1+z0+p1        # this has maximum value of 3
    condition[condition <= 0.0] = 4.0
    condition[condition <= 3.0] = 0.0
    return condition/4


def Thermal2DSG(xyVals, p) :
    # Three 2D gaussian peaks with specified displacements. Widths are assumed to be the same:
    #   MinOne (X,Y) Central position   : p[0][0,1]
    #   Zero (X,Y) Central position     : p[1][0,1]
    #   PluOne (X,Y) Central position   : p[2][0,1]
    #   (X,Y) width / 2                 : p[3][0,1]
    #   (-1|0|1) peak height            : p[4][0,1,2]
    #   xyVals[0] and xyVals[1] are x and y coordinates where stuff is calculated
    xm = p[0]
    xz = p[1]
    xp = p[2]
    xw = p[3][0]
    yh = p[3][1]
    total = p[4][0]*np.exp(-1*(xyVals[0]-xm[0])**2/(2*xw**2)-1*(xyVals[1]-xm[1])**2/(2*yh**2))
    total = total + p[4][1]*np.exp(-1*(xyVals[0]-xz[0])**2/(2*xw**2)-1*(xyVals[1]-xz[1])**2/(2*yh**2))
    total = total + p[4][2]*np.exp(-1*(xyVals[0]-xp[0])**2/(2*xw**2)-1*(xyVals[1]-xp[1])**2/(2*yh**2))
    return total

def puckThermal2DSG(xyVals, p) :
    # Three 2D gaussian peaks with specified displacements and elliptical masks. Widths are assumed to be the same:
    #   MinOne (X,Y) Central position   : p[0][0,1]
    #   Zero (X,Y) Central position     : p[1][0,1]
    #   PluOne (X,Y) Central position   : p[2][0,1]
    #   (X,Y) thermal width / 2         : p[3][0,1]
    #   (-1|0|1) peak height            : p[4][0,1,2]
    #   (X,Y) width / 2 of masks        : p[5][0,1]
    #   xyVals[0] and xyVals[1] are x and y coordinates where stuff is calculated
    thermalDist = Thermal2DSG(xyVals, p[0:5])
    pp=p[0:3]+(p[5],)
    masks = puck2DSG(xyVals, pp)
    thermalDist = thermalDist*masks
    return thermalDist


def absline(xyVals, p) :
    # absolute value of a line
    #   intercept   : p[0]
    #   slope       : p[1]
    return np.abs(p[0] + p[1]*xyVals[0])

def avdcrossing(xyVals, p) :
    # avoided crossing by using absolute value of a line + a p[2] gap from the x-axis
    #   intercept       : p[0]
    #   slope           : p[1]
    #   gap             : p[2]
    return np.sqrt((p[0] + p[1]*xyVals[0])**2 + p[2]**2.0)

#=================================================================================
# I don't understand these two functions?!.. with sVals?
def Mask2D(xyVals, p) :
    # Three 2D pucks with specified displacements. Widths are assumed to be the same.
    # To be used with sVals
    #   X border        : p[0]
    #   Y border        : p[1]
    x = xyVals[2]
    total = puck(xyVals, (x[0],p[0],x[1],p[1]))
    total = total + 1.0
    total[total > 1.5] = 100000.0
    return total
    
def Mask2DSG(xyVals, p) :
    # Three 2D pucks with specified displacements. widths are assumed the same. for use with sVals:
    #   X border        : p[0]
    #   Y border        : p[1]
    xm = xyVals[2]
    xz = xyVals[3]
    xp = xyVals[4]
    total = puck(xyVals, (xm[0],p[0],xm[1],p[1]))
    total = total + puck(xyVals, (xz[0],p[0],xm[1],p[1]))
    total = total + puck(xyVals, (xp[0],p[0],xp[1],p[1]))
    total = total + 1.0
    total[total > 1.5] = 100000.0
    return total

#=================================================================================

# Consider rewriting the three following functions below. Seems like they only work for 1D cases
def interpfunc(xyVals, p):
    # 1D interpolation function
    #   offset of mf = -1 cloud : p[0]
    #   offset of mf = +1 cloud : p[1]
    return xyVals[1](xyVals[0]+p[0]) + xyVals[2](xyVals[0]) + xyVals[3](xyVals[0]+p[1])

def minimize_offset(xyVals, p):
    totalOD = interpfunc((xyVals[0], xyVals[1], xyVals[2], xyVals[3]), p)
    p_guess = (0,0.5,0.0,40.0)
    res = curve_fit_qgas(TF_only, p_guess, totalOD, (xyVals[0],), full_output=1)
    return res[1]

def shift_cloud(xyVals, p):
    # shift clouds around
    #   shift of mf = -1 cloud    : p[0]
    #   shift of mf = +1 cloud    : p[1]
    #   offset for TF function    : p[2]
    #   amplitude for TF          : p[3]
    #   location for TF           : p[4]
    #   radius for TF             : p[5]
    #   xyVals[0] are the coordinates
    #   xyVals[1|2|3] are [Minus one|Zero|Plus one] amplitudes
    intm = interp1d(xyVals[0],xyVals[1],bounds_error = False, fill_value = 0.0)
    intz = xyVals[2]
    intp = interp1d(xyVals[0],xyVals[3],bounds_error = False, fill_value = 0.0)
    return intm(xyVals[0]+p[0])+intz+intp(xyVals[0]+p[1]) - TF_only((xyVals[0],), (p[2],p[3],p[4],p[5]))
    
#==============================================================================
# 
# Functions to execute the fit to IanStyleFits
#       Need to modify curve_fit_qgas and FitFunctionForOptimize to perform fitting with some fixed parameters.
#       Right now, fitting with fixed parameters are performed by modifying func
#==============================================================================

def curve_fit_qgas(func, p_guess, zVals, xyVals, sVals = None, **kw):
    ''' curve_fit_qgas extends the operation of the scipy curve fit
        to more naturally deal with higher dimensional functions
        
        func : the function to be fit, formed as func(xyVals, p)
            xyVals:  is a tuple or list or array of arrays, each 
            xyVals[0] ... xyVals[N] is an array of coordinates
            so for example compare the N-dimensional function 
            evaluated at xyVals[0][q] ... xyVals[N][q] to zVals[q]
            each of the xyVals[p] can be a matrix, as they will 
            be .ravel()'ed to make 1D arrays internally.
            
            p : is the array of parameters
        
        p_guess : the initial guess of parameters
        
        zVals : data        
        
        xyVals : coordinates where data is known (as described in func)
        
        sVals : uncertainties on each point, defaults to 1

        kw : additional parameters to pass to scipy.optimize.leastsq
    '''
 
    #==============
    # make sure that zVals, xyVals[], and svals are numpy arrays
    # If they are already, these functions will still make local copies
    # this may be slow, but in a fit the fit loop will be the problem
    #==============

    p_guessInt = np.array(p_guess, dtype=float);

    zValsInt = np.array(zVals, dtype=float);
    
    xyValsInt = [np.array(x, dtype=float) for x in xyVals];
    
    # If no uncertainties were passed set them to 1
    if (sVals is not None):
        sValsInt = np.array(sVals, dtype=float);
    else:
        sValsInt = None;
                
    # construct the desired fit function
    func = FitFunctionForOptimize(func, zValsInt, xyValsInt, sVals = sValsInt);

    # Remove full_output from kw, otherwise we're passing it in twice.
    return_full = kw.pop('full_output', False)
    res = scipy.optimize.leastsq(func, p_guessInt, full_output=1, **kw)
    (popt, pcov, infodict, errmsg, ier) = res

    if ier not in [1,2,3,4]:
        msg = "Optimal parameters not found: " + errmsg
        # print msg
        return p_guessInt, p_guessInt  #just sends back random stuff to keep the program running
        #raise RuntimeError(msg) # this aborts the batchrun program nicely

    # Generate covariance matrix
    if (zValsInt.size > p_guessInt.size) and pcov is not None:
        s_sq = (func(popt)**2).sum()/(zValsInt.size-p_guessInt.size);
        pcov = pcov * s_sq;
    else:
        pcov = np.inf;

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov

# constructor function
def FitFunctionForOptimize(func, zVals, xyVals, sVals = None):
    '''
    Returns a function which takes "p" as a parameter for scipy.optimize
    func is the function to be fit of the form gaussian2D(xyVals, p), 
    xyVals = (xVals, yVals, ...)
    is a tuple of values where zVals are defined
    sVals is the array of uncertainties if it passed
    '''

    # define the internal function to return the sequence of residuals as a 1D array.
    
    # If no uncertainties were passed, proceed as if they are equal to 1
    if (sVals is None):
        def OptFunct(p):
            return ((func(xyVals, p) - zVals)).ravel();
    else:
        def OptFunct(p):
            return ((func(xyVals, p) - zVals)/sVals).ravel();

    # numpy.ravel(a, order='C')
    # Return a flattened array.
    # A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
    
    return OptFunct;
    
#================================================================================
# 
# Some generic functions --> I am not sure that CorrectOD is used at all for now?
#
#================================================================================

# Correct the optical depth for saturation intensity and Doppler shift effects.
def CorrectOD(ODRaw, CountsRaw, PulseTime, ISatCounts, tau):
    """
    Gives the corrected optical depth, given:
    ODRaw           The measured OD
    CountsRaw       Number of counts w/o atoms
    PulseTime       Imaging PulseDuration
    ISatCounts      ISat in count
    tau             recoil time (19 us for 40K, 42 us for 87Rb)    
    """
    IoverIsat = CountsRaw / ISatCounts;
    ODCorrect = -np.log(( IoverIsat * np.exp(-ODRaw) + 1.0)/(IoverIsat  + 1.0));
    ODCorrect += -( 1.0 / ( IoverIsat * np.exp(-ODRaw) + 1.0) - 1.0 / ( IoverIsat + 1.0) );
    ODCorrect *= (1.333)*(PulseTime/tau)**2;
    ODCorrect += ODRaw + IoverIsat * (1.0 - np.exp(-ODRaw));
    return ODCorrect;