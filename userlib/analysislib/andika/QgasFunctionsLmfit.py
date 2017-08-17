"""
Project   : python BEC analysis
Filename  : QgasFunctionsLmfit

Created on  : 2014 Dec 22 11:10
Author      : aputra
QgasFunctions: Mathematical functions in quantum gas experiments, to be mainly used in data-fitting.
               This module is intended to do fitting using lmfit package with lmfit.Model() parameter class.
"""

import numpy as np
import lmfit
from scipy.linalg import expm
from scipy.special import gamma

# === 1D functions ===

def TF_only(xyVals, offset, A, x0, R) :
    # A Thomas Fermi peak with:
    #   Constant Background             : offset
    #   Peak height above background    : A
    #   Central position                : x0
    #   Radius                          : R
    condition = (1 - ((xyVals-x0)/R)**2)
    condition[condition < 0.0] = 0
    return offset + A*(condition**(3.0/2.0))

def TF_onlyChopped(xyVals, offset, A, x0, R, xc) :
    # A Thomas Fermi peak with:
    #   Constant Background             : offset
    #   Peak height above background    : A
    #   Central position                : x0
    #   Radius                          : R
    #   Chopped distance from center    : xc
    condition = (1 - ((xyVals-x0)/R)**2)

    # if (xc> (-2)*R) and (xc< 2*R):
    xbound = [ x0 + np.sign(xc)*(R-np.abs(xc)), x0 + np.sign(xc)*R]
    xbound = np.sort(xbound)
    for i in range(len(xyVals)):
        if xyVals[i] > (xbound[0]) and xyVals[i] < (xbound[1]):
            condition[i] = 0

    condition[condition < 0.0] = 0
    return offset + A*(condition**(3.0/2.0))

def gaussian(xyVals, offset, A, x0, sigma) :
    # A gaussian peak with:
    #   Constant Background             : offset
    #   Peak height above background    : A
    #   Central position                : x0
    #   Standard deviation              : sigma
    return offset + A*np.exp(-1*(xyVals-x0)**2/(2*sigma**2))

def lorentzian(xyVals, offset, A, x0, sigma) :
    # A lorentzian peak with:
    #   Constant Background             : offset
    #   Peak height above background    : A
    #   Central position                : x0
    #   Full-Width at Half-Maximum      : 2*sigma
    return offset + A/(1.0+((xyVals-x0)/sigma)**2)

def line(xyVals, intercept, slope) :
    # A linear fit with:
    #   y-Intercept         : intercept
    #   slope or gradient   : slope
    return intercept + slope*xyVals

def quadratic(xyVals, a, b, c):
    # A quadratic fit with a, b, c coefficients
    return a*xyVals**2 + b*xyVals + c
	
def power(xyVals, offset, A, x0, k) :
    # A power law fit with:
    #   Amplitude           : A
    #   Position center     : x0
    #   Power               : k
    #   Background Constant : offset
    return A*(xyVals-x0)**k + offset
	
	
def absline(xyVals, intercept, slope) :
    # absolute value of a line
    #   y-intercept : intercept
    #   slope       : slope
    return np.abs(intercept + slope*xyVals)

def avdcrossing(xyVals, intercept, slope, gap) :
    # avoided crossing by using absolute value of a line + a y-gap from the x-axis
    return np.sqrt((intercept + slope*xyVals)**2 + gap**2.0)

	
def stepfn(xyVals, initial, final, mid_time, rise_time):
    exp_term = -4*np.log(3)*(xyVals-mid_time)/rise_time	
    return (final - initial)/(1 + np.exp(exp_term)) + initial
	
def sine(xyVals, f, A, offset, phi):
    return offset + A*np.sin(2*np.pi*f*xyVals+phi)
	
def sine_decay(xyVals, f, A, offset, phi, tc):
    return offset + A*np.exp(-xyVals/tc)*np.sin(2*np.pi*f*xyVals+phi)
	
def sigmoid(xyVals, base, A, xhalf, rate):
    return base + A/(1+np.exp(-(xyVals-xhalf)/rate))
	
def gen_rabi(xyVals, det, fR, A, offset, phi):
    return offset + A*fR**2/(det**2 + fR**2)*np.cos(2*np.pi*np.sqrt(det**2+fR**2)*xyVals+phi)

def poisson_dist(xyVals, A, lamb, xscale):
    return A*(lamb**(xyVals/xscale)) * np.exp(-lamb)/gamma((xyVals/xscale)+1)

# === Numerical function ===
def lattice_evol(xyVals, Vlatt, n_diff):
    #Vlatt = 10.0;
    #xyVals = np.linspace(0, 50e-6, 101);
    tt = 2*np.pi*3678*xyVals; # this is in SI units where time is input in 's'
    Norder = 9;
    Nsize = 2*Norder+1;
    c_diff = np.zeros((Nsize,1));
    c_diff[Norder] = 1;
    NN = np.linspace(-Norder,Norder,Nsize);
    onHH = 4*(NN)**2+Vlatt/2;
    offHH = Vlatt/4*np.ones((Nsize-1,1));
    Ham = np.diag(onHH) + np.diagflat(offHH, 1) + np.diagflat(offHH, -1)
    Ptotal = np.zeros((Nsize,len(tt)));

    for it in range(0,len(tt)):
        t_evol = tt[it];
        ct = np.dot(expm(-1j*Ham*t_evol),c_diff);
        Ptotal[:,[it]] = np.absolute(ct)**2;
    #plt.plot(tt,Ptotal[Norder+1,:])

    return Ptotal[Norder+n_diff,:]

# === 2D functions ===

def TF_only2D(xyVals, offset, A, x0, y0, R_x, R_y) :
    # A 2D gaussian peak with:
    #   Constant Background             : offset
    #   Peak height above background    : A
    #   X Central position              : x0
    #   X width                         : R_x
    #   Y Central position              : y0
    #   Y width                         : R_y
    condition = (1.0 - ((xyVals[0]-x0)/R_x)**2.0 - ((xyVals[1]-y0)/R_y)**2.0)
    condition[condition < 0.0] = 0.0
    return (offset + A*(condition**(3.0/2.0))).ravel()

def gaussian2D(xyVals, offset, A, x0, y0, sigma_x, sigma_y) :
    # A 2D gaussian peak with:
    #   Constant Background             : offset
    #   Peak height above background    : A
    #   X Central position              : x0
    #   X Standard deviation            : sigma_x
    #   Y Central position              : y0
    #   Y Standard deviation            : sigma_y
    model2D = offset + A*np.exp(-1*(xyVals[0]-x0)**2/(2*sigma_x**2)-1*(xyVals[1]-y0)**2/(2*sigma_y**2))
    return model2D.ravel()

def gaussian2DSG(xyVals, offset, A_m, x0_m, y0_m, A_z, x0_z, y0_z, A_p, x0_p, y0_p, sigma_x, sigma_y) :
    # Three 2D gaussian peaks with specified displacements.
    # Thermal widths and heights are assumed to be the same:
    #   MinOne (X,Y) Central position   : p[0][0,1]
    #   Zero (X,Y) Central position     : p[1][0,1]
    #   PluOne (X,Y) Central position   : p[2][0,1]
    #   (X,Y) width / 2                 : p[3][0,1]
    #   (-1|0|1) peak height            : p[4][0,1,2]
    #   xyVals[0] and xyVals[1] are x and y coordinates where stuff is calculated

    total = offset + A_m * np.exp(-1*(xyVals[0]-x0_m)**2/(2*sigma_x**2)-1*(xyVals[1]-y0_m)**2/(2*sigma_y**2))
    total = total  + A_z * np.exp(-1*(xyVals[0]-x0_z)**2/(2*sigma_x**2)-1*(xyVals[1]-y0_z)**2/(2*sigma_y**2))
    total = total  + A_p * np.exp(-1*(xyVals[0]-x0_p)**2/(2*sigma_x**2)-1*(xyVals[1]-y0_p)**2/(2*sigma_y**2))
    return total.ravel()

def puck(xyVals, x0, ax, y0, by) :       # an ellipse mask
    # returns 1.0 inside the ellipse. 0.0 everywhere else:
    #   X Central position  : x0
    #   X major axis / 2    : ax
    #   Y Central position  : y0
    #   Y minor axis / 2    : by
    condition = (1.0 - ((xyVals[0]-x0)/ax)**2.0 - ((xyVals[1]-y0)/by)**2.0)
    condition[condition < 0.0] = 0.0
    condition[condition > 0.0] = 1.0
    return condition.ravel()

def puck2DSG(xyVals, x0_m, y0_m, x0_z, y0_z, x0_p, y0_p, ax, by) :
    # three ellipse masks which share common major and minor axes
    # returns 1.0 within three ellipses. 0.0 everywhere else:
    m1 = (1.0 - ((xyVals[0]-x0_m)/ax)**2.0 - ((xyVals[1]-y0_m)/by)**2.0)
    m1[m1 < 0.0] = 0.0
    z0 = (1.0 - ((xyVals[0]-x0_z)/ax)**2.0 - ((xyVals[1]-y0_z)/by)**2.0)
    z0[z0 < 0.0] = 0.0
    p1 = (1.0 - ((xyVals[0]-x0_p)/ax)**2.0 - ((xyVals[1]-y0_p)/by)**2.0)
    p1[p1 < 0.0] = 0.0
    condition = m1+z0+p1        # this has maximum value of 3
    condition[condition <= 0.0] = 4.0
    condition[condition <= 3.0] = 0.0
    return (condition/4).ravel()


# === Fitting ====

def FitQgas(xyVals, data, func, params):
    model = lmfit.Model(func)
    if data.ndim > 1:
        data = data.ravel()
    result = model.fit(data, params, xyVals=xyVals)
    return result