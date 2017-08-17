
from __future__ import division
from lyse import *
from time import time
from matplotlib import pyplot as plt
from common.traces import *
from common.plot_pretty import plot_pretty
from common.wave_process import single_wave, multi_wave
import os
import pandas as pd
import numpy as np

print '\nRunning %s' % os.path.basename(__file__)
t = time()

@plot_pretty(marker='o', 
             markeredgecolor='royalblue', 
             markerfacecolor='lightsteelblue', 
             markeredgewidth=1.5, 
             markersize=6, 
             linestyle='-', 
             color='midnightblue', 
             linewidth=1.5)
def plot_waves(wave_x, wave_y, show_errorbar=False, is_data_fit=False):
    if show_errorbar:
        plotline, caplines, barlinecols = plt.errorbar(wave_x['wave_data'], 
                                                       wave_y['wave_data'],
                                                       xerr = 0*wave_x['wave_uncer'], 
                                                       yerr = wave_y['wave_uncer'], 
                                                       capsize=0, 
                                                       ls='None')
        line_objects = plotline, caplines, barlinecols
    elif is_data_fit:
        line_objects = plt.plot(wave_x['wave_data'], 
                                wave_y['wave_data'], 
                                label= wave_y['wave_name'] + ' ' + str(wave_y['wave_fit_params']))
    else:
        line_objects = plt.plot(wave_x['wave_data'], 
                                wave_y['wave_data'], 
                                label= wave_y['wave_name'])    
    plt.xlabel(wave_x['wave_name'], fontsize=8)
    plt.ylabel(wave_y['wave_name'], fontsize=8)
    plt.title('Plot: ' + wave_y['wave_name'] + ' vs ' + wave_x['wave_name'], fontsize=8)
    return line_objects, show_errorbar, is_data_fit
    
try:
    # Call waves from main dataframe
    if True:
        _RunNo_ = single_wave(wave_name='RunNo').get_single()
        number_1 = single_wave(wave_name='N_(1,0)', single_shot_routine='get_OD').normalize(norm='val', val=1e5)
        run_no = multi_wave(_RunNo_, number_1).build_RunNo_wave()
        time = single_wave(wave_name='short_TOF').normalize(norm='val', val=1.0)
        pkOD =  single_wave(wave_name='pkOD',  single_shot_routine='get_OD').get_single()
        temp = single_wave(wave_name='temperature', single_shot_routine='get_OD').normalize(norm='val', val=1e-9)
        width = single_wave(wave_name='x_gauss_width', single_shot_routine='get_OD').get_single()
        com = single_wave(wave_name='x_gauss_center', single_shot_routine='get_OD').get_single()
        
    # 1D Fourier analysis
    if False:
        number_psd = single_wave(wave_name='N_(2,2)', single_shot_routine='get_OD').psd()
        f_space = single_wave(wave_name='TimeHold').reciprocal_space()
        
    # Spin projection analysis
    if False:
        pulse_time = single_wave(wave_name='time_microwave_pulse').get_single()
        #number_m1 = single_wave(wave_name='N_(1,-1)', single_shot_routine='get_OD').get_single()
        #number_0 = single_wave(wave_name='N_(1,0)', single_shot_routine='get_OD').get_single()
        number_2 = single_wave(wave_name='N_(2,2)', single_shot_routine='get_OD').get_single()
        
        oneplustwo = multi_wave(number_2, number_1).add_waves()
        #total_number = multi_wave(onepluszero, number_m1).add_waves()
        #pop_ratio_0 = multi_wave(total_number, number_0).take_ratio()
        pop_ratio_2 = multi_wave(oneplustwo, number_2).take_ratio()
        pop_ratio_1 = multi_wave(oneplustwo, number_1).take_ratio()
    
    # Fits 
    if False:
        try:
            fit_x1, fit_y1= multi_wave(dipole, temp).fit_wave(trace='linear')
        except:
            fit_x1, fit_y1 = run_no, number_1
    # Plot multi-plots on demand
    if True:
        fig1 = plt.figure(1, figsize=(8, 18), frameon=False)
        fig1.add_subplot(211)   
        plot_waves(time, com, show_errorbar=False, is_data_fit=False)
        #plot_waves(fit_time, fit_COM, show_errorbar=False, is_data_fit=True)
        fig1.add_subplot(212)     
        plot_waves(time, width, show_errorbar=False, is_data_fit=False)
        #plot_waves(fit_time, fit_number_2, show_errorbar=False, is_data_fit=True) 
        
        plt.tight_layout()
    
    # Plot single-plot on demand
    if False:
        fig2 = plt.figure(2, figsize=(8, 6), frameon=False)
        fig2.add_subplot(111)    
        try:
            plot_waves(time, com, show_errorbar=False, is_data_fit=False)
            #plot_waves(fit_x1, fit_y1, show_errorbar=False, is_data_fit=True)
            #plot_waves(fit_freq, fit_number_1, show_errorbar=False, is_data_fit=True)  
        except:
            plot_waves(run_no, number_1, show_errorbar=False, is_data_fit=False)
        plt.show()
        plt.tight_layout()
    
    # Show
    fig1 = plt.gcf()
    fig2 = plt.gcf()
    
    
    print '\n ********** Successful **********\n\n'
except Exception as e:
    print '%s' %e 
    print '\n ********** Not Successful **********\n\n'

    
""" Wishlist """

    #avg_wave_x, avg_wave_y = average_wave(('None', 'get_OD', 'None'), ('sequence_index', 'y_gauss_center', 'short_TOF'),
    #                                      as_wave_x='short_TOF', wave_avg='sequence_index', common='y_gauss_center')
    
# - Parse fit_params guess all the way into traces.py
