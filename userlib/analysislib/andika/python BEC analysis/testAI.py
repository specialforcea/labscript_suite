"""
Project   : python BEC analysis
Filename  : testAI

Created on  : 2015 Jan 07 12:26
Author      : aputra
"""
import QgasFileIO as iogas
import matplotlib.pyplot as plt
from pylab import *
import numpy as np

filename = "/Users/andika28putra/Dropbox/December/17/SC1_17Dec2014_2000.ibw";
SCtime, SCscope = iogas.LoadSC(filename)


plt.figure()
plt.plot(SCtime[2], SCscope[2], 'm.-')
plt.plot(SCtime[3], SCscope[3], 'rx-')

plt.show()