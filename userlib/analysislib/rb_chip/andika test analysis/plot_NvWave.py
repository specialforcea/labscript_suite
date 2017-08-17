from __future__ import division
from lyse import *
from pylab import *
from analysislib.spinor.aliases import *
import os
from time import time
import pandas as pd
import numpy as np

print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()

N0 = df['get_ODandika', 'Ntotal']
N0 = N0
N1 = (N0).tolist()
Ndim = len(N1)

Nfrac1 = df['get_ODandika', 'Nfrac1']
Nfrac3 = df['get_ODandika', 'Nfrac3']
#print Nfrac2
std_dev = np.std(N1)

pulsetime = df['TimeRamanPulse']
offBias = df['dTpzShim']

runNo = np.linspace(0, Ndim-1, Ndim)


Nmax = max(N1)
Nmin = min(N1)

plot(pulsetime*1000000,N0,'bo')
#plot(pulsetime*1000000,Nfrac1,'ro')
#plot(runNo,N0,'bo')
#plot(offBias, N1, 'bo')
axis([0, 1300, 100, 71000])


xlabel('PulseTime (in us)', fontsize=12)
#xlabel('dTpzShim', fontsize=14)
ylabel('N total', fontsize=12)
#title('Map Depth (100)')
tight_layout()
show()