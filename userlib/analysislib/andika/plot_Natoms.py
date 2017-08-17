
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

N0 = df['get_OD', 'N']
N0 = N0
N1 = (N0).tolist()
Ndim = len(N1)

std_dev = np.std(N1)

P_f = df['raman_wavelength']

runNo = np.linspace( 0,Ndim-1, Ndim)
sizeN1 = len(P_f)
Nmax = max(N1)
Nmin = min(N1)
plot(runNo, N1, 'bo')
# errorbar(P_f, N1, yerr=std_dev/2, fmt='bo')
axis([0, 15, -1000, 1.5*Nmax])
xlabel('Run No', fontsize=18)
ylabel('Atom counts')
#title('Map Depth (150)')
tight_layout()
show()

