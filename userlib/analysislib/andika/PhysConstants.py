"""
Project   : python BEC analysis
Filename  : PhysConstants

Created on  : 2014 Dec 16 13:50
Author      : aputra

A module containing physical constants and functions. All units in SI, otherwise it will be stated explicitly
"""

import numpy as np
from scipy.integrate import odeint

hbar = 1.05457e-34
m87Rb = 1.44316072e-25
kBoltz = 1.3806503e-23
Rb_D2lambda = 780.241e-9                    # Rubidium D2 transition wavelength, from 5S1/2 to 5P3/2
sigma_o = (3*(Rb_D2lambda**2))/(2*np.pi)    # On resonance scattering cross section for absorption imaging
aRb_swave = 5.313e-9                        # Rb s-wave scattering length in meter
#omegashort =
#omegalong =


def NumberCalcAbsOD(FitImage,ScaleFactor):

    pixelarea = ((ScaleFactor)*1e-6)**2     # pixel area in meter
    # If imaging is calibrated, 2DImage(x,y) = sigma_o * n(x,y) with n = column density
    n = FitImage/sigma_o
    N = np.sum(n*pixelarea)
    return N

def NumberCalcSum(FitImage,ScaleFactor):
    pixelarea = ((ScaleFactor))**2     # pixel area in meter
    # If imaging is calibrated, 2DImage(x,y) = sigma_o * n(x,y) with n = column density
    N = np.sum(FitImage*pixelarea)
    return N
	
# Atom number after TOF from TF fitting, based on Castin & Dum's paper
def NumberCalcTOF_TFfit(TFRadiusLong,TFRadiusShort,omegashort,omegalong,TOFTime):

    # Returns number N in absolute values and chemical potential mu in Hz
    # use TF radii input into function in units of um = 1e-6 meters, omega is 2*pi*f! Include your 2*pi's!

    t = np.linspace(0.0,TOFTime,500)
    yinit = np.array([1.0, 1.0, 0.0, 0.0]) # initial values
    yout = odeint(cdum_deriv, yinit, t, args = (omegashort,omegalong))

#   Calculate Castin-Dum parameters for axially elongated symmetric case:
#     tau = omegashort*TOFTime;
#     epsilon = omegalong/omegashort;

# This only applies for elongated traps with high aspect ratio:
#     lambda_t = (sqrt(1+tau**2));
#     lambda_z = (1+(epsilon**2)*(tau*atan(tau)-log(sqrt(1+tau**2))));

    lambda_t = yout[len(yout[:,1])-1,1];
    lambda_z = yout[len(yout[:,1])-1,2];

# Use Castin-Dum equation to get original radii in meters for axially symmetric trap
    TFRadiusLong_insitu = (TFRadiusLong/lambda_z)*1e-6;
    TFRadiusShort_insitu = (TFRadiusShort/lambda_t)*1e-6;

    omega_avg = (omegashort*omegashort*omegalong)**(1.0/3.0);
    mu_avg = 0.5*m87Rb*(1.0/3.0)*((omegashort*TFRadiusShort_insitu)**2 + (omegashort*TFRadiusShort_insitu)**2 + (omegalong*TFRadiusLong_insitu)**2);
    N = (1/15.0/aRb_swave)*np.sqrt(hbar/m87Rb/omega_avg)*(2*mu_avg/hbar/omega_avg)**(5.0/2.0);

    # Set chemical potential mu to Hz units
    mu = mu_avg/(hbar*2*np.pi);

    return (N, mu, TFRadiusShort_insitu/1e-6, TFRadiusLong_insitu/1e-6)


def cdum_deriv(y, t, omegashort, omegalong): # return derivatives of the array y based on Castin Dum's
    # Castin Dum equations:
    # y[0] = lambda_perp
    # y[1] = lambda_z
    # y[2] = d lambda_perp / dt
    # y[3] = d lambda_z / dt
    return np.array([y[2], y[3], omegashort**2/y[0]**3/y[1], omegalong**2/y[0]**2/y[1]**2])


def ThermalTemp(thermWidth, thermHeight, TOF):
    # Calculate quantum gas temperature based on thermal fit after TOF
    # thermWidth and thermHeight in um
    thermTemp = np.sqrt((kBoltz*2*TOF**2)/m87Rb)*np.sqrt(thermWidth*thermWidth*thermHeight*(1e-6)**3);
    return thermTemp*(1e6)  # return temperature in uK