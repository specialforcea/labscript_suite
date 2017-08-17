from __future__ import division
from lyse import *
from pylab import *
from analysislib.spinor.aliases import *
import os
from time import time
import pandas as pd
import numpy as np
from dika_fitting import *

print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()

N0 = df['get_ODandika', 'Ntotal']
N0 = N0
N1 = (N0).tolist()
Ndim = len(N1)

subdf = df.sequences(last=1)

Nfrac1 = df['get_ODandika', 'Nfrac1'].get_values()
Nfrac2 = df['get_ODandika', 'Nfrac2'].get_values()
Nfrac3 = df['get_ODandika', 'Nfrac3'].get_values()
Natoms = df['get_ODandika', 'N3'].get_values()
std_dev = np.std(N1)

pulsetime = np.transpose(df['TimeRamanPulse'].get_values())[0]*1000000;

offBias = np.transpose(df['dTpzShim'].get_values())[0];
#offBias = df['TpyShimEnd']
runNo = np.linspace(0, Ndim-1, Ndim)


Nmax = max(N1)
Nmin = min(N1)


#pars_guess = np.array([5000.0, 1.0, 0.0, 0.0, 100.0e-6])

#params, u_params = fit_exp(pulsetime, Nfrac3);
#print params
#xfit = linspace(min(pulsetime),max(pulsetime),len(pulsetime)*3)
#yfit = expfn(xfit, params[0],params[1],params[2])

#params, u_params = fit_sine_decay(pulsetime, Nfrac3);
#print params
#xfit = linspace(min(pulsetime),max(pulsetime),len(pulsetime)*3)
#yfit = sine_decay(xfit, params[0],params[1],params[2],params[3],params[4])

#params, u_params = fit_sigmoid(offBias, Nfrac2);
#print params
#xfit = linspace(min(offBias),max(offBias),len(offBias)*3)
#yfit = sigmoid(xfit, params[0],params[1],params[2],params[3])


#plot(offBias,Nfrac2,'bo')
#plot(runNo,Nfrac3,'bo-')
#noRaman, = plot(pulsetime[:26], Nfrac3[:26], 'bo',label="no Raman")
#Raman1, = plot(pulsetime[26:52], Nfrac3[26:52], 'rx',label="Raman 1 only")
#Raman2, = plot(pulsetime[52:], Nfrac3[52:], 'gv',label="Raman 2 only")

sort_idx = np.argsort(pulsetime);

Nfm1, = plot(pulsetime,Nfrac1,label="mf = -1", linestyle = "None", marker = "o")
Nf0, = plot(pulsetime,Nfrac2,label="mf = 0", linestyle = "None", marker = "v")
#Nf0, = plot(xfit,yfit,label="mf = 0")
Nfp1, = plot(pulsetime,Nfrac3,label="mf = 1", linestyle = "None", marker = "x")

#Nfm1, = plot(offBias,Nfrac2,label="mf = -1", linestyle = "None", marker = "o")
#Nf0, = plot(offBias,Natoms/100000,label="mf = -1", linestyle = "None", marker = "o")
#Nf0, = plot(xfit,yfit,label="mf = 0")
#Nf0, = plot(offBias,Nfrac2,label="mf = 0", linestyle = "None", marker = "v")
#Nfp1, = plot(offBias,Nfrac3,label="mf = 1", linestyle = "None", marker = "o")

#legend(handles=[Nfm1,Nf0,Nfp1])

#axis([0, 110, -0.1, 1.1])

#xlabel('dTpzShim', fontsize=12)
xlabel('PulseTime (in us)', fontsize=12)
#xlabel('Time hold (in s)', fontsize=12)
ylabel('mf 0 fraction', fontsize=12)
#title('Map Depth (100)')
tight_layout()
show()