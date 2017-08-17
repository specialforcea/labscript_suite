from __future__ import division
from lyse import *
import os
from time import time
import matplotlib.pyplot as plt
import QgasFunctionsLmfit as Qfunc
import matplotlib.gridspec as gridspec

#import pandas as pd
import numpy as np
import lmfit


print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()

pulsetime = np.transpose(df['TimeRamanPulse'].get_values())[0]*1000000;

pOneX = df['RabiAnalysis','pOneX'].get_values();
pOneY = df['RabiAnalysis','pOneY'].get_values();
zeroX = df['RabiAnalysis','zeroX'].get_values();
zeroY = df['RabiAnalysis','zeroY'].get_values();

mOne = df['RabiAnalysis', 'mOne'].get_values()
zero = df['RabiAnalysis', 'zero'].get_values()
pOne = df['RabiAnalysis', 'pOne'].get_values()

Nfrac1 = pOne/(zero+pOne)#+mOne)
Nfrac0 = zero/(zero+pOne)#+mOne)


pguess = lmfit.Parameters()
pguess.add('offset', value = 0.5, vary = False)
pguess.add('A', value = 0.5, vary = False)
pguess.add('f', value = 4/pulsetime.max())
pguess.add('phi', value = 0.05)
pguess.add('tc', value = 2*pulsetime.max())

fit_par = Qfunc.FitQgas(pulsetime, Nfrac1, Qfunc.sine_decay, pguess)
xfit = np.linspace(min(pulsetime),max(pulsetime),len(pulsetime)*3)
fitdata = Qfunc.sine_decay(xfit, **fit_par.values)
print(fit_par.fit_report())

plt.figure(4)
gs = gridspec.GridSpec(4, 1)

ax1 = plt.subplot(gs[0])
ax1.plot(pulsetime,Nfrac1,label="mf = 1", linestyle = "None", marker = "o")
ax1.plot(pulsetime,Nfrac0,label="mf = 0", linestyle = "None", marker = "x")
ax1.plot(xfit,fitdata,label="fit mf=1")
ax1.set_xlabel('Pulse Time (in us)', fontsize=12)
ax1.set_ylabel('Atom fraction', fontsize=12)
#ax1.legend(loc='upper right', fontsize=8)

ax2 = plt.subplot(gs[1], sharex = ax1)
ax2.plot(pulsetime,pOneX,label="mf = 1", linestyle = "None", marker = "o")
ax2.plot(pulsetime,zeroX,label="mf = 0", linestyle = "None", marker = "x")
ax2.set_xlabel('Pulse Time (in us)', fontsize=12)
ax2.set_ylabel('x-center', fontsize=12)

ax3 = plt.subplot(gs[2], sharex = ax1)
ax3.plot(pulsetime,pOneY,label="mf = 1", linestyle = "None", marker = "o")
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
plt.plot(pulsetime,distance, label="total dist", linestyle = "None", marker = "o")
plt.plot(pulsetime,zeroX-pOneX, label="x-dist", linestyle = "None", marker = "x")
#plt.plot(pulsetime,zeroY-pOneY, label="y-dist", linestyle = "None", marker = "v")
plt.xlabel('Pulse Time (in us)', fontsize=12)
plt.ylabel('CloudDistance', fontsize=12)
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
