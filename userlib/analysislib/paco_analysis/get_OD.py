# Takes RAW arrays and returns calculated OD for given shot
# along with the best fit (between gaussian and TF) for ROI.

from __future__ import division
from time import time
from scipy.ndimage import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd
import numpy as np
import numexpr as ne
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from lyse import *
import fit_table

from common.OD_handler import ODShot, ODslice
from common.traces import bimodal1D
from analysislib.common.get_raw_images import get_raw_images

# Parameters
pixel_size = 5.6e-6/3.44# Divided by Magnification Factor
            # 5.6e-6/5.33 for z in situ                        # Yuchen and Paco: 08/19/2016
            #5.6e-6/3.44 for z TOF                           # Yuchen and Paco: 08/19/2016
            #5.6e-6/2.72 for x-in situ                        # Paco: 05/06/2016
sigma0 = 3*(780.24e-9)**2/(2*np.pi)  # Atomic cross section (resonant absorption imaging)
save = True
fit_slices = True

# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()

# Run method
run = Run(path)

# Methods
def print_time(text):
    print 't = %6.3f : %s' % ((time()-t), text)
 
def raw_to_OD(fpath):
    reconstruction_group = 'reconstruct_images'
    atoms, probe, bckg = get_raw_images(fpath)
    rchi2_item = (reconstruction_group, 'reconstructed_probe_rchi2')
    df = data(fpath)
    if rchi2_item in df and not np.isnan(df[rchi2_item]):
        with h5py.File(fpath) as f:
            if reconstruction_group in f['results']:
                probe = run.get_result_array('reconstruct_images', 'reconstructed_probe')
                bckg = run.get_result_array('reconstruct_images', 'reconstructed_background')
    div = np.ma.masked_invalid((atoms - bckg)/(probe - bckg))
    div = np.ma.masked_less_equal(div, 0.)
    Isat = 127. # Paco: 05/12/2017 **only for insitu **
    another_term = 1.*(probe-atoms)/(Isat)
    alpha = 0.92    #  Paco: 05/16/2017 **only for insitu **
    calculated_OD = np.array(-alpha*np.log(div) + another_term)
    return np.matrix(calculated_OD)

# Main
try:
    with h5py.File(path) as h5_file:
        if '/data' in h5_file:
            print_time('Calculating OD...')
            # Get OD, ROI, BCK
            _OD_ = raw_to_OD(path)
            OD = ODShot(_OD_)
            F, mF, _ROI_, BCK_a =  OD.get_ROI(sniff=False, get_background=False) 
            _, _, ROIcoords, _ = np.load(r'C:\labscript_suite\userlib\analysislib\paco_analysis\ROI_temp.npy')
            point1, point2 = ROIcoords
            x1, y1 = point1
            x2, y2 = point2
            ROI = ODShot(np.matrix(np.array(_ROI_)**(3)))
            BCK = np.nanmean(BCK_a)*np.ones(_ROI_.shape)
            if True:
                N = (np.sum((_ROI_-BCK)/sigma0)*pixel_size**2)
            else:
                N = np.nan
            if save:
                run.save_result( 'pkOD', (np.nanmax(_ROI_.astype(np.float16))))
                run.save_result(('N_(' + str(F) +',' +str(mF)+')'), N)
                run.save_result('y_COM', np.abs(ROI.COM_2D(0, 0)[0]))
            # Slices
            print_time('Slice...')
            #xcolOD, x_ax = OD.slice_by_segment_OD(coord_a=np.array([221, 75]), coord_b=np.array([236, 600]))
            xcolOD, x_ax = np.mean(np.array(_OD_[213:217, 100:570]), axis=0), np.linspace(0, 470, 470)
            ycolOD, y_ax = OD.slice_by_segment_OD(coord_a=np.array([40, 354]), coord_b=np.array([430, 354]))
            y_ax=y_ax[::-1]   # Reverse to match gravity
            # Fits
            if fit_slices:
                _xslice_, _yslice_ = ODslice(slice_OD=xcolOD, slice_axis=x_ax), ODslice(slice_OD=ycolOD, slice_axis=y_ax)
                run.save_result('SNR', (np.nanmean(xcolOD[410:440])/np.std(xcolOD[290:320])))
                try:
                    x_gauss_pars, x_dense_gauss, x_gaussian_fit =_xslice_.fit_gauss()
                    x_tf_pars, x_dense_tf, x_TF_fit = _xslice_.fit_pure_thomas_fermi()
                    x_bimodal_pars, x_dense_bimodal, x_bimodal_fit =_xslice_.fit_bimodal()
                    fit_x_success = True
                    print 'x slice fit'
                    fit_table.get_params(x_bimodal_pars)
                    if save:
                        run.save_result('x_gauss_width', np.abs(x_gauss_pars[2]*pixel_size/(1e-6)))
                        run.save_result('x_gauss_amp', np.abs(x_gauss_pars[0]-x_gauss_pars[3]))
                        run.save_result('x_gauss_center', x_gauss_pars[1]*pixel_size/1e-6)
                        thermal = bimodal1D(x_ax, x_bimodal_pars[0], 0., x_bimodal_pars[2], x_bimodal_pars[3],  0., x_bimodal_pars[5])
                        fraction = (np.sum(xcolOD) - np.sum(thermal))/np.sum(xcolOD)
                        run.save_result('x_condensate_fraction', fraction)
                        run.save_result('temperature', (1.44e-25*(x_bimodal_pars[3]*pixel_size)**2)/(2*1.38e-23*(24.72e-3**2)))
                except Exception as e:
                    fit_x_success = False
                    print 'Fit of x slice unsuccessful, %s' %e
                try:
                    y_gauss_pars, y_dense_gauss, y_gaussian_fit = _yslice_.fit_gauss()
                    y_tf_pars, x_dense_tf, y_TF_fit = _yslice_.fit_pure_thomas_fermi()
                    y_bimodal_pars, y_dense_bimodal, y_bimodal_fit =_yslice_.fit_bimodal()
                    fit_y_success = True
                    print 'y slice fit'
                    fit_table.get_params(y_gauss_pars)
                    if save:
                        run.save_result('y_gauss_center', np.abs(216-y_gauss_pars[1]*pixel_size/1e-6))
                except Exception as e:
                    fit_y_success = False
                    print 'Fit of y slice unsuccessful, %s' %e
                if save:
                    run.save_result('integrated_linOD', _xslice_.integrate())
            else:
                fit_x_success, fit_y_success = False, False
            # Display OD, slices and N
            figOD = plt.figure(figsize=(8, 5), frameon=False)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[4,1])
            
            plt.subplot(gs[2])
            number_display = r'N = %d' % N
            plt.text(0.4, 0.6, number_display, ha='center', va='top', fontsize=18)
            plt.gca().axison = False

            plt.subplot(gs[1])
            im0= plt.imshow(np.array(_OD_)**(3/3), vmin= -0.35, vmax =0.5, cmap='viridis', aspect='equal', interpolation='none')
            #plt.axvline(349, color='r', linewidth=2.5)
            #plt.axvline(x2, color='r', linewidth=0.5)
            #plt.axhline(np.abs(OD.COM_2D(0, 0)[0]), color='r', linewidth=1.5)
            #plt.axhline(203, color='r', linewidth=2.5)
            grid_divider = make_axes_locatable(plt.gca())
            cax = grid_divider.append_axes("right", "5%", pad="3%")
            plt.colorbar(im0, cax=cax)
            plt.title('OD')
            # X slice
            plt.subplot(gs[3])
            plt.step(x_ax*pixel_size/1e-6, xcolOD, 'k', linewidth=0.5)
            if fit_x_success:
                #pass
                plt.plot(x_dense_bimodal*pixel_size/1e-6, 0*x_bimodal_fit, 'r')
                plt.plot(x_dense_gauss*pixel_size/1e-6, x_gaussian_fit, 'b--')
            plt.xlabel('$x \,[\mu m]$', fontsize=15)
            plt.ylabel('OD')
            plt.title('x_slice')
            #plt.xlim(np.amin(x_ax), np.amax(x_ax))
            # Y slice
            plt.subplot(gs[0])
            plt.step(ycolOD, y_ax*pixel_size/1e-6, 'k', linewidth=0.5)
            if fit_y_success:
                plt.plot(y_gaussian_fit, y_dense_gauss*pixel_size/1e-6, 'r')
            plt.xlabel('OD')
            plt.ylabel('$y \, [\mu m]$', fontsize=15)
            plt.title('y_slice')
            #plt.ylim(np.amin(y_ax), np.amax(y_ax))

            plt.tight_layout()
            plt.show()
        else:
            print_time('Unsuccessful...')
            raise Exception( 'No image found in file...' )
        print '\n ********** Successful **********\n\n'
except Exception as e:
    print '%s' %e +  os.path.basename(path)
    print '\n ********** Not Successful **********\n\n'