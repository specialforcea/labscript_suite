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

offBias = np.transpose(df['dTpzShim'].get_values())[0];

mOne = df['lyseAnalysis', 'mOne'].get_values()
zero = df['lyseAnalysis', 'zero'].get_values()
pOne = df['lyseAnalysis', 'pOne'].get_values()


Nfrac0 = zero/(zero+pOne)


pguess = lmfit.Parameters()
pguess.add('base', value = 0, vary=False)
pguess.add('A', value = 1, vary=False)
pguess.add('xhalf', value = offBias.max())
pguess.add('rate', value = abs(offBias.max()))

fit_par = Qfunc.FitQgas(offBias, Nfrac0, Qfunc.sigmoid, pguess)
xfit = np.linspace(min(offBias),max(offBias),len(offBias)*3)
fitdata = Qfunc.sigmoid(xfit, **fit_par.values)
print fit_par.values
print(fit_par.fit_report())

pguess = lmfit.Parameters()
pguess.add('initial', value = 0, vary=False)
pguess.add('final', value = 1, vary=False)
pguess.add('mid_time', value = offBias.max())
pguess.add('rise_time', value = abs(offBias.max()))
fit_parfn = Qfunc.FitQgas(offBias, Nfrac0, Qfunc.stepfn, pguess)
fitdatafn = Qfunc.stepfn(xfit, **fit_parfn.values)



plt.plot(offBias,Nfrac0,label="mf = 0", linestyle = "None", marker = "o")
plt.plot(xfit,fitdata, linestyle = ':')
plt.plot(xfit,fitdatafn, linestyle = '--')

plt.xlabel('dTpzShim', fontsize=12)
plt.ylabel('Nfrac0', fontsize=12)

plt.show()