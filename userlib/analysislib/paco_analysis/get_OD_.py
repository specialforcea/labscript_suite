# Takes RAW arrays and returns calculated OD for given shot
# along with the best fit (between gaussian and TF) for ROI.

from __future__ import division
from lyse import *
from pylab import *
from common.fit_gaussian_2d import fit_2d
from common.traces import *
from spinor.aliases import *
from time import time
from scipy.ndimage import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from common.OD_handler import ODShot
import os
import pandas as pd
import numpy as np
import numexpr as ne
import matplotlib.gridspec as gridspec
import fit_table
from analysislib.common.get_raw_images import get_raw_images

# Parameters
pixel_size = 5.6e-6/5.33# Divided by Magnification Factor
            # 5.6e-6/5.33 for z in situ                        # Yuchen and Paco: 08/19/2016
            #5.6e-6/3.44 for z TOF                           # Yuchen and Paco: 08/19/2016
            #5.6e-6/2.72 for x-in situ                        # Paco: 05/06/2016
sigma0 = 3*(780e-9)**2/(2*3.14)
# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()

# Load dataframe
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
    another_term = 0 # (probe-atoms)/(Isat)
    alpha = 1.0
    calculated_OD = np.array(-alpha*np.log(div) + another_term)
    return np.matrix(calculated_OD)
                
# Main
try:
    #plt.xkcd()
    with h5py.File(path) as h5_file:
        if '/data' in h5_file:
            print_time('Calculating OD...')
        # Get OD
            _OD_ = raw_to_OD(path)
            print_time('Get OD...')
            OD = ODShot(_OD_)
            F, mF, _ROI_, BCK_a =  OD.get_ROI(sniff=False, get_background=False) 
            _, _, ROIcoords, _ = np.load(r'C:\labscript_suite\userlib\analysislib\paco_analysis\ROI_temp.npy')
            point1, point2 = ROIcoords
            x1, y1 = point1
            x2, y2 = point2
            ROI = ODShot(_ROI_)
            BCK = np.mean(BCK_a)*np.ones(_ROI_.shape)
            run.save_result( 'pkOD', (np.amax(_ROI_.astype(float16))))
        # Compute number
            if True: #stored == "z-TOF":
                N = (np.sum((_ROI_-BCK)/sigma0)*pixel_size**2)
                print 0.2*pixel_size**2/sigma0
                run.save_result(('N_(' + str(F) +',' +str(mF)+')'), N)
            else:
                N = 0
        # Display figure with shot, slices and fits and N
            fig = figure(figsize=(8, 5), frameon=False)
            gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[4,1])
            subplot(gs[2])
            str = r'N = %.0f' % N
            text(0.4, 0.6, str, ha='center', va='top',fontsize=18)
            gca().axison = False
            tight_layout()
        # OD and ROI display
            subplot(gs[1])
            im0= imshow(_OD_, vmin= -0.0, vmax = 0.4, cmap='viridis', aspect='equal', interpolation='none')
            #axvline(x1, color='r')
            #axvline(x2, color='r')
            #axhline(y1, color='r')
            #axhline(y2, color='r')
            divider = make_axes_locatable(gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            colorbar(im0, cax=cax)
            title('OD')
            tight_layout()
            # Raw data is displayed and if fits are unsuccesful only show raws.
        # Slice
            print_time('Slice and fit...')
            xcolOD, x_ax = OD.slice_by_segment_OD(coord_a=np.array([170, 100]), coord_b=np.array([170, 600]))
            ycolOD, y_ax = OD.slice_by_segment_OD(coord_a=np.array([50, 322]), coord_b=np.array([300, 322]))
            y_ax=y_ax[::-1]
        # Raw data is displayed and if fits are unsuccesful only show raws.
            # Gaussian 1D
            x_gaussian_par, dx_gaussian_par = fit_gaussian(x_ax, xcolOD)
            y_gaussian_par, dy_gaussian_par = fit_gaussian(y_ax, ycolOD)
            run.save_result('x_gauss_width', np.abs(x_gaussian_par[2]*pixel_size/(1e-6*4*np.log(2))))
            run.save_result('y_gauss_width', np.abs(y_gaussian_par[2]*pixel_size/(1e-6*4*np.log(2))))
            run.save_result('2dwidth', np.sqrt(x_gaussian_par[2]**2+y_gaussian_par[2]**2))
            run.save_result('gauss_amp', np.abs(x_gaussian_par[0]-x_gaussian_par[3]))
            run.save_result('x_gauss_center', np.where(xcolOD == np.amax(xcolOD))[0][0])
            run.save_result('y_gauss_center', (480-y_gaussian_par[1])*5.6e-6)
            run.save_result('integrated_linOD', np.sum(xcolOD))
            print 'x Gaussian fit'
            #fit_table.get_params(dx_gaussian_par)
            print 'y Gaussian fit'
            #fit_table.get_params(dy_gaussian_par)
            x_gaussian_fit = gaussian(x_ax, x_gaussian_par[0], x_gaussian_par[1], x_gaussian_par[2], x_gaussian_par[3])
            y_gaussian_fit = gaussian(y_ax, y_gaussian_par[0], y_gaussian_par[1], y_gaussian_par[2], y_gaussian_par[3])
            if (x_gaussian_fit is not None or y_gaussian_fit is not None):
                # Thomas Fermi 1D
                print_time('Gauss fit successful for x and y')
                x_tf_par, dx_tf_par = fit_thomas_fermi(x_ax, xcolOD)
                y_tf_par, dy_tf_par = fit_thomas_fermi(y_ax, ycolOD)
                print 'x Thomas Fermi fit'
                fit_table.get_params(dx_tf_par)
                print 'y Thomas Fermi fit'
                fit_table.get_params(dy_tf_par)
                x_tf_fit = thomas_fermi(x_ax, x_tf_par[0], x_tf_par[1], x_tf_par[2], x_tf_par[3])
                y_tf_fit = thomas_fermi(y_ax, y_tf_par[0], y_tf_par[1], y_tf_par[2], y_tf_par[3])
                run.save_result('ATF', (x_tf_par[2]**2+y_tf_par[2]**2))
                if(x_tf_fit is not None or y_tf_fit is not None):
                    print_time('TF fit successful for x and y')
                    subplot(gs[3])
                    plot(x_ax, xcolOD, 'b', x_ax, x_gaussian_fit, 'r', x_ax, (x_tf_fit), 'g')
                    xlabel('xpos (um)')
                    ylabel('OD')
                    title('x_slice')
                    #axis([0, 600, -0.4, 2.0])
                    tight_layout()
                    subplot(gs[0])
                    plot(ycolOD, y_ax, y_gaussian_fit, y_ax, 'r', y_tf_fit, y_ax, 'g')
                    xlabel('OD')
                    ylabel('ypos (um)')
                    title('y_slice')
                    #axis([-0.4, 2.0, 600, 0])
                    tight_layout()
                    show()
                    run.save_result('TF_width', np.abs(2*y_tf_par[2]*pixel_size/(1e-6)))
                    #omega_par = np.sqrt(3*N*1.05e-34*2*pi*30e3*5.3e-9/(1.44e-25*(x_tf_par[2]*1.7/(2*1e6))**3))/(2*pi)
                    #run.save_result('freq_long', omega_par)
                else:
                    raise Exception ('Can only do Gaussian fit')
                    subplot(gs[3])
                    plot(x_ax, xcolOD, 'b', x_ax, x_gaussian_fit, 'r')
                    xlabel('xpos (um)')
                    ylabel('OD')
                    title('x_slice')
                    #axis([0, 600, -0.4, 2.0])
                    tight_layout()
                    subplot(gs[0])
                    plot(ycolOD, y_ax, y_gaussian_fit, y_ax, 'r')
                    xlabel('OD')
                    ylabel('ypos (um)')
                    title('y_slice')
                    #axis([-0.4, 2.0, 600, 0])
                    tight_layout()
                    show()
                run.save_result('gauss_amp', np.abs(y_gaussian_par[0]))
            else:
                raise Exception ('Can\'t fit')
                print_time('Gauss fit unsuccessful for x or y')
                subplot(gs[3])
                plot(x_ax, xcolOD)
                xlabel('xpos (um)')
                ylabel('OD')
                title('x_slice')
                #axis([0, 600, -0.4, 2.0])
                tight_layout()
                subplot(gs[0])
                plot(ycolOD, y_ax)
                xlabel('OD')
                ylabel('ypos (um)')
                title('y_slice')
                #axis([-0.4, 2.0, 600, 0])
                tight_layout()
                show()
        else:
            print_time('Unsuccessful...')
            raise Exception( 'No image found in file...' )
        print '\n ********** Successful **********\n\n'
except Exception as e:
    print '%s' %e +  os.path.basename(path)
    print '\n ********** Not Successful **********\n\n'