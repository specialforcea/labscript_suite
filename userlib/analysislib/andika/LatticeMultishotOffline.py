from __future__ import division

# For LYSE
#from lyse import *

import os
from time import time
import matplotlib.pyplot as plt
import QgasFunctionsLmfit as Qfunc
import matplotlib.gridspec as gridspec

import numpy as np
import lmfit
import h5py

# Folder and file locations
dataFolder = "//129.6.128.236/Labscript_Shared/Experiments/rb_chip/"
sequenceFile = "BECspinDepLatt"
year = "2016"
month = "04"
day = "19"
seqIndex = "0020"
idStart = 00
idEnd = 40

# TODO: consider using the os module to get filename from the folder.
foldername = dataFolder + sequenceFile + "/" + year + "/" + month + "/" + day + "/" +seqIndex + "/"
fileheader = year + "_" + month + "_" + day + "_" + sequenceFile + "_"
t = time()


idRange = np.linspace(idStart,idEnd,41);

pulsetime = np.zeros_like(idRange);
pOneX = np.zeros_like(idRange);
pOneY = np.zeros_like(idRange);
zeroX = np.zeros_like(idRange);
zeroY = np.zeros_like(idRange);
mOne = np.zeros_like(idRange);
zero = np.zeros_like(idRange);
pOne = np.zeros_like(idRange);
ii = 0;
for idFile in idRange:
    filename = foldername + fileheader + str("%.2d" % idFile) + ".h5"
    print filename
    with h5py.File(filename) as h5_file:
        pulsetime[ii] = h5_file['globals'].attrs['TimeRamanPulse']
       # pOneX[ii] = h5_file['results/LatticeAnalysis'].attrs['pOneX'];
        #pOneY[ii] = h5_file['results/LatticeAnalysis'].attrs['pOneY'];
        #zeroX[ii] = h5_file['results/LatticeAnalysis'].attrs['zeroX'];
        #zeroY[ii] = h5_file['results/LatticeAnalysis'].attrs['zeroY'];
        mOne[ii] = h5_file['results/LatticeAnalysis'].attrs['mOne'];
        zero[ii] = h5_file['results/LatticeAnalysis'].attrs['zero'];
        pOne[ii] = h5_file['results/LatticeAnalysis'].attrs['pOne'];
    ii = ii+1;




pOne = (pOne > 0) * pOne;
mOne = (mOne > 0) * mOne;
    
Nfrac1 = pOne/(zero+pOne+mOne)
Nfrac0 = zero/(zero+pOne+mOne)
Nfracm1 = mOne/(zero+pOne+mOne)
Ntotal = (zero+pOne+mOne)
'''
Ndim = len(Nfrac1)
runNo = np.linspace(0, Ndim-1, Ndim)
plt.figure(3)
plt.plot(runNo,Nfrac1,'bo-')
'''

pguess = lmfit.Parameters()
#pguess.add('offset', value = 0.5, vary=True)
#pguess.add('A', value = 0.5, vary=True)
#pguess.add('f', value = 8/pulsetime.max())
#pguess.add('phi', value = 0.05)
#pguess.add('tc', value = 2*pulsetime.max())
pguess.add('Vlatt', value = 8)
pguess.add('n_diff', value = 1, vary=False)

fit_par = Qfunc.FitQgas(pulsetime, Nfrac1, Qfunc.lattice_evol, pguess)
xfit = np.linspace(min(pulsetime),max(pulsetime),len(pulsetime)*3)
fitdata = Qfunc.lattice_evol(xfit, **fit_par.values)
print(fit_par.fit_report())

pguess.add('n_diff', value = 0, vary=False)

fit_par = Qfunc.FitQgas(pulsetime, Nfrac0, Qfunc.lattice_evol, pguess)
fitdata0 = Qfunc.lattice_evol(xfit, **fit_par.values)
print(fit_par.fit_report())

plt.figure(4)
gs = gridspec.GridSpec(4, 1)

ax1 = plt.subplot(gs[0])
ax1.plot(pulsetime,Nfrac1,label="+1 order", linestyle = "None", marker = "o")
ax1.plot(pulsetime,Nfrac0,label="0 order", linestyle = "None", marker = "x")
ax1.plot(pulsetime,Nfracm1,label="-1 order", linestyle = "None", marker = "v")
ax1.plot(xfit,fitdata0,label="fit 0th order")
ax1.plot(xfit,fitdata,label="fit 1st order")
ax1.set_xlabel('Pulse Time (in us)', fontsize=12)
ax1.set_ylabel('Atom fraction', fontsize=12)
ax1.legend(loc='upper right', fontsize=8)

ax2 = plt.subplot(gs[1], sharex = ax1)
#ax2.plot(pulsetime,pOneX,label="mf = 1", linestyle = "None", marker = "o")
ax2.plot(pulsetime,zeroX,label="mf = 0", linestyle = "None", marker = "x")
ax2.set_xlabel('Pulse Time (in us)', fontsize=12)
ax2.set_ylabel('x-center', fontsize=12)

ax3 = plt.subplot(gs[2], sharex = ax1)
#ax3.plot(pulsetime,pOneY,label="mf = 1", linestyle = "None", marker = "o")
ax3.plot(pulsetime,zeroY,label="mf = 0", linestyle = "None", marker = "x")
ax3.set_xlabel('Pulse Time (in us)', fontsize=12)
ax3.set_ylabel('y-center', fontsize=12)

distance = np.sqrt( (pOneY-zeroY)**2 + (pOneX-zeroX)**2)
ax4 = plt.subplot(gs[3], sharex = ax1)
ax4.plot(pulsetime,distance, label="total dist", linestyle = "None", marker = "o")
ax4.plot(pulsetime,zeroX-pOneX, label="x-dist", linestyle = "None", marker = "x")
ax4.set_xlabel('Pulse Time (in us)', fontsize=12)
ax4.set_ylabel('CloudDistance', fontsize=12)
ax4.legend(loc='upper right', fontsize=8)
plt.tight_layout()


plt.figure(5)
fit2 = Qfunc.lattice_evol(xfit,8.87, 2)
plt.plot(xfit,fit2)