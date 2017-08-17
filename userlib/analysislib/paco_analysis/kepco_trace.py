
from __future__ import division
from lyse import *
from pylab import *
from analysislib.common.fit_gaussian_2d import fit_2d
from analysislib.common.traces import *
from analysislib.spinor.aliases import *
from time import time
from scipy.ndimage import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd
import numpy as np
import numexpr as ne
import matplotlib.gridspec as gridspec
from analysislib import fit_table


# Parameters
pixel_size = 5.6e-6/4  # Divided by Magnification Factor

# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()
def print_time(s):
    print 't = %6.3f : %s' % ((time()-t), s)
	
# Run dataframe
run = Run(path)

x_trace, y_trace = run.get_trace('trace_fluxgate')

plot(x_trace, y_trace, 'r')
xlabel('t (s)')
ylabel('output (V)')
title('fluxgate_trace')
tight_layout()