
from __future__ import division
from lyse import *
from time import time
from matplotlib import pyplot as plt
from common.traces import *
import os
import numpy as np

print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()

def pull_wave(method, wave_name):
    wave_attrs = {}
    if method == 'None':
        wave = df[wave_name]
    else:
        wave = df[method, wave_name]
    wave_arr = np.array(wave)
    wave_attrs['wave_size'] = np.amax(wave_arr.shape)
    wave_attrs['wave_std'] = np.nanstd(wave_arr)
    wave_attrs['wave_min'] ,wave_attrs['wave_max'] = np.amin(wave_arr), np.amax(wave_arr)
    wave_attrs['wave_name'] = wave_name
    return wave_arr, wave_attrs

def plot_waves(method_1, wavename_1, method_2, wavename_2):
    wave_2, wave_2_attrs = pull_wave(method_2, wavename_2)
    if wavename_1 == 'RunNo':
        wave_1_attrs = ({'wave_name': wavename_1, 'wave_min': 0.0,
                        'wave_max': wave_2_attrs.get('wave_size'), 
                        'wave_size':wave_2_attrs.get('wave_size')})
        wave_1 = np.linspace(0, wave_1_attrs.get('wave_max'), wave_1_attrs.get('wave_size'))
    else:
        wave_1, wave_1_attrs = pull_wave(method_1, wavename_1)
    # Build figure
    plt.plot(wave_1, wave_2, 'bo')
    #plt.errorbar(wave_1, wave_2, yerr=wave_2_attrs.get('wave_std'), fmt='bo')
    plt.axis([wave_1_attrs.get('wave_min'), wave_1_attrs.get('wave_max'), 
             2*wave_2_attrs.get('wave_min'), 2*wave_1_attrs.get('wave_max')])
    plt.xlabel(wave_1_attrs.get('wave_name'), fontsize=18)
    plt.ylabel(wave_2_attrs.get('wave_name'), fontsize=18)
    plt.title('Plot: ' + wave_2_attrs.get('wave_name') + ' vs ' + wave_1_attrs.get('wave_name'))
    plt.tight_layout()
    plt.show()

def fit_wave(x_wave, y_wave, fitting_curve):
    wave_fit_command =
try:
    plot_waves('None', 'RunNo', 'get_OD', 'N')
    #fit_wave(wave1, wave2, 'gaussian')
except:
    pass