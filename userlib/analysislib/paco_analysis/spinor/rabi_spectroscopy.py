from __future__ import division
from lyse import *
from pylab import *
from common import *
from alkali.rubidium87 import *
import pandas as pd

# Flags
save_csv = False
save_plot = True
show_plot = True
solve_field = True
shaded_plot = False
uncertainties = True
show_fit = True

# Parameters
state_list = ['a', 'b', 'c']
# state_list = ['c', 'f']
minimum_points = 4
plot_units = 'MHz'

# Constants
red = (0.67,0,0)
blue = (0,0,0.67)
green = (0,0.67,0)

# Pulse duration
pulse_duration = 'sp_pulse_duration'
# pulse_duration = 'mwave_pulse_duration_1'

# x-axis
x_global = 'sp_center_freq'
# x_global = 'mwave_freq_1'

# y-axis
# y_global = ('roi_atoms','Fz')
y_global = ('roi_atoms', 'Fz_fit')

# y-axis uncertainty
u_y_global = u_global(y_global)

# Columns group
cols_group = 'roi_atoms'

if spinning_top:
    df = data()
else:
    try:
        df = data()
    except:
        df = pandas.read_pickle('20140409_spinor.pickle')

# only use the most recent sequences
subdf = df.sequences(last=1)

# use particular sequences
# subdf = df.sequences('20140409T163121')

# omit NaNs
cols = subdf.multicols([x_global, y_global[-1], u_y_global[-1]], how='any')
subdf = subdf.dropnans(cols)

if len(subdf):
    # fitting
    t_pulse = subdf[pulse_duration][0]
    # x_units = Run(subdf['filepath'][0]).get_units()[x_global]
    x_units = 'MHz'

    # compose data arrays
    xdata = subdf[x_global]*si[x_units]
    ydata = subdf[y_global]
    
try:
    u_ydata = subdf[u_y_global]
    if any(u_ydata != 0) and uncertainties:
        uncertainties = True
        # Clip u_ydata between smallest non-negligible uncertainty and 0.5
        u_ydata = around(u_ydata, 4)
        u_ydata = clip(u_ydata, u_ydata[u_ydata > 0].min(), 0.5)
    else:
        u_ydata = None
        uncertainties = False
except:
    u_ydata = None
    uncertainties = False

# sequence names
seq_str = subdf.sequence_strings()
plot_title = subdf.sequence_string()

def print_unc(name):
    return name + ' = ' + str(eval('format_unc(%s,u_%s)' % (name, name)))

# Zeeman splitting of states specified by a list e.g. ['a', 'b', ..] 
def splitting(states, B):
    # average splitting between states in Hz for B in Gauss
    state_ints = [ord(x)-ord('a') for x in sort(states)]
    energies = array(rubidium_87_S12_state.energy_eigenstates(B*1e-4)[0])
    Ediff = mean(energies[state_ints][1:]-energies[state_ints][:-1])/(2*pi*hbar)
    return Ediff
    
zero_field_splitting = splitting(state_list, 0)

if len(subdf) >= minimum_points:
    print '\nRunning %s on %s' % (os.path.basename(__file__), seq_str)
    try:
        params, u_params, covariance = fit_rabi(xdata-zero_field_splitting, ydata, t_pulse, u_ydata=u_ydata, maxfev=10000, return_covariance=True)
        params += array([zero_field_splitting, 0, 0, 0])
        params_names = ['center_frequency', 'Rabi_frequency', 'Rabi_amplitude', 'Rabi_offset']
        center_frequency, Rabi_frequency, Rabi_amplitude, Rabi_offset = params
        u_center_frequency, u_Rabi_frequency, u_Rabi_amplitude, u_Rabi_offset = u_params
        def fitfn(f, f0, fR, A, c):
            return rabi(t_pulse, f, f0, fR, A, c)      
        fit_success = True
        # raise RuntimeError('Bypass Rabi fit')
    except RuntimeError:
        try:
            params, u_params = fit_lorentzian(xdata, ydata)
            params_names = ['center_frequency']
            center_frequency = params[1]  
            u_center_frequency = u_params[1]
            fitfn = lorentzian
            fit_success = True
        except:            
            fit_success = False        
            params_names = []
    
    if solve_field and fit_success:
        from scipy.optimize import fsolve
        from scipy.misc import derivative

        dB = 1e-3
        B0 = (center_frequency-zero_field_splitting)/derivative(lambda B: splitting(state_list, B), dB, dx=dB)
        try:
            B = fsolve(lambda B: splitting(state_list, B) - center_frequency, B0)[0]
            u_B = abs(u_center_frequency/derivative(lambda B: splitting(state_list, B), B, dx=dB))
            params_names.append('B')
        except:
            solve_field = False

    if show_plot:
        figure('Rabi spectroscopy')
        xdata_p = linspace(min(xdata), max(xdata), 1024)
        if uncertainties:
            errorbar(xdata/si[plot_units], ydata, yerr=u_ydata, fmt = 'o', ms=5, mfc=red, mec=red, ecolor=red)
        else:
            plot(xdata/si[plot_units], ydata, marker='o', ls='None', color=red)
        if fit_success and show_fit:
            if shaded_plot:
                probability, extent = model_shaded_uncertainty(fitfn, xdata_p, params, covariance)
                extent[0] /= si[plot_units]
                extent[1] /= si[plot_units]
                imshow(1-probability, origin='lower-left', aspect='auto',  extent=extent, cmap=cm.bone)
            else:
                plot(xdata_p/si[plot_units], fitfn(xdata_p, *params), color=red)
        xmin, xmax, ymin, ymax = axis(xmin=(1.05*min(xdata)-0.05*max(xdata))/si[plot_units], 
                                      xmax=(1.05*max(xdata)-0.05*min(xdata))/si[plot_units],
                                      ymin=-1.05, ymax=1.05)
        xlabel(x_global + ' (%s)' % plot_units)
        ylabel(y_global)
        title(plot_title)
        params_names_print = []
        for name in params_names:
            if eval('abs(%s) > abs(u_%s)' % (name, name)):
                params_names_print.append(name)
        if fit_success:
            summary_text =  '\n'.join(map(print_unc, params_names_print))
        else:
            summary_text = 'Fit failed'
        print summary_text
        text(0.01*xmin+0.99*xmax, 0.01*ymin+0.99*ymax, summary_text, 
             horizontalalignment='right', verticalalignment='top', 
             backgroundcolor=(1,1,1,0.5))
        if save_plot:
            path = os.path.split(subdf['filepath'][-1])[0]
            png_path = os.path.join(path, seq_str[-1] + '_Rabi.png')
            savefig(png_path)
        show()

    # package it into a pandas dtype
    if fit_success:
        series_names = ravel(map(lambda name: (name, 'u_' + name), params_names))
        fit_series = seriesify(*map(eval,series_names))
               
    # save CSV of desired data
    if save_csv:
        # compose column list we want to save
        cols = [tuplify(subdf, x_global)] + subdf.multicols(cols_group)

        # save CSV
        if save_csv:
            path = os.path.split(subdf['filepath'][-1])[0]
            csv_path = os.path.join(path,seq_str[-1] + '_Rabi.csv')
            subdf[cols].to_csv(csv_path)
            print 'Saved ' + csv_path