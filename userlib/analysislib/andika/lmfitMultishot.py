from __future__ import division
from lyse import *
import os
from time import time
import matplotlib.pyplot as plt
import QgasFunctionsLmfit as Qfunc

#import pandas as pd
import numpy as np
import lmfit


print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()

pulsetime = np.transpose(df['pulsetime'].get_values())[0];

mOne = df['lyseAnalysis', 'mOne'].get_values()
zero = df['lyseAnalysis', 'zero'].get_values()
pOne = df['lyseAnalysis', 'pOne'].get_values()

Nfrac1 = pOne/(zero+pOne+mOne)




plt.plot(pulsetime,Nfrac1,label="mf = 0", linestyle = "None", marker = "o")
plt.plot(xfit,fitdata)
plt.xlabel('pulsetime', fontsize=12)
plt.ylabel('Nfrac1', fontsize=12)

plt.show()