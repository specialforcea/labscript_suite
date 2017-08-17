# Takes RAW arrays and returns calculated OD for given shot
# along with the best fit (between gaussian and TF) for ROI.

from __future__ import division
from lyse import *
from pylab import *
from analysislib.common.fit_gaussian_2d import fit_2d
from analysislib.spinor.aliases import *
from time import time
from scipy.ndimage import *
from scipy.optimize import *
import os
import pandas as pd
import numpy as np
import numexpr as ne

# Parameters
pixel_size = 5.6e-6/3.15  # Divided by Magnification Factor

# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()
def print_time(s):
    print 't = %6.3f : %s' % ((time()-t), s)
	
# Main
run = Run(path)
image_group = 'imagesXY_2_Flea3'

# Methods and Functions

def raw_to_OD(name):
     with h5py.File(path) as h5_file:
        abs = ma.masked_invalid(array(h5_file['data'][image_group][name])[0])
        probe = ma.masked_invalid(array(h5_file['data'][image_group][name])[1])
        bckg = ma.masked_invalid(array(h5_file['data'][image_group][name])[2])
        return matrix(-log((abs - bckg)/(probe - bckg)))

def get_ROI_guess(multi_ROI):
     if multi_ROI:
        x0 = 0
     else:
         with h5py.File(path) as h5_file:
            x0 = int(h5_file['globals/lyse_wise'].attrs['ROI_box_x0'])
            xf = int(h5_file['globals/lyse_wise'].attrs['ROI_box_xf'])
            y0 = int(h5_file['globals/lyse_wise'].attrs['ROI_box_y0'])
            yf = int(h5_file['globals/lyse_wise'].attrs['ROI_box_yf'])
            multiple_ROI = h5_file['globals/lyse_wise'].attrs['multiple_ROI']
            return x0, xf, y0, yf

def get_ROI_box(ODshot, ODuntrans, threshold_OD):
    
    x_off = ODshot.shape[0] - ODuntrans.shape[0]
    y_off = ODshot.shape[1] - ODuntrans.shape[1]
    
    artif_backg = np.zeros(ODshot.shape)
    OD_thrs = ma.masked_where(ODshot < threshold_OD, ODshot) + artif_backg  
    
    x0, xf, y0, yf = get_ROI_guess(False)
    
    ROI_box = OD_thrs[x0:xf+x_off:1, y0:yf+y_off:1]
    ROI_COM = measurements.center_of_mass(ROI_box)
    
    max_guess = measurements.maximum_position(array(ROI_box))
    xx = int(max_guess[0])
    yy = int(max_guess[1])
    
    ROI_abs_center = ([xx, yy])#([x0 + (xf-x0)/2, y0 + (yf-y0)/2])

    return ROI_box, ROI_COM, ROI_abs_center
    
#### WARNING: CURRENTLY ONLY WORKS FOR SINGLE ROI ####

#def set_ROI(ODshot, threshold_OD):
	# # First stage, coarse (MAX) search
    # artif_backg = np.zeros(ODshot.shape)
    # OD_thrs = ma.masked_where(ODshot < threshold_OD, ODshot) + artif_backg  
    # first_guess = measurements.maximum_position(array(OD_thrs))
    # x0 = int(first_guess[0])
    # y0 = int(first_guess[1])
    # x0max = int(ODshot.shape[0])  
    # y0max = int(ODshot.shape[1])
    # ROI_hint = OD_thrs[int(x0-0.1*x0max):int(x0+0.1*x0max):1, int(y0-0.1*y0max):int(y0+0.1*y0max):1]
	
	# # Second stage, finer (COM) search
    # com_guess = measurements.center_of_mass(ROI_hint)
    # xf = int(com_guess[0]+(x0-0.1*x0max))
    # yf = int(com_guess[1]+(y0-0.1*y0max))
    # OD_COM = xf, yf
    # ROI_TH = OD_thrs[(xf-0.1*x0max):(xf+0.1*x0max):1, (yf-0.1*y0max):(yf+0.1*y0max):1]
    # ROI_COM = measurements.center_of_mass(ROI_TH)
    # ROI_f = ODshot[(xf-0.1*x0max):(xf+0.1*x0max):1, (yf-0.1*y0max):(yf+0.1*y0max):1]  
	
    #return ROI_f, ROI_COM

def slice_width_avgd(ODshot, pixel_width, center, bound, xslice):
    if (pixel_width % 2 == 0):
        raise Exception('The argument pixel_width'
		                'should be odd so that it'
                        'covers an evenly split range')
    else:
        xpos = center[0]
        ypos = center[1]
        xbound = bound[1]
        ybound = bound[0]
        if xslice:
            xuppr = int(xpos + (pixel_width-1)/2)
            xlowr = int(xpos - (pixel_width-1)/2)
            empty_slice = [0] * xbound #648
            for index in range(xlowr, xuppr+1):
                indx_slice = reshape(ODshot[index,:], xbound) #648
                empty_slice = empty_slice + indx_slice
            avgd_slice = empty_slice/pixel_width
            pixels = np.arange(0, xbound, 1) #648
        elif not xslice:
            yuppr = int(ypos + (pixel_width-1)/2)
            ylowr = int(ypos - (pixel_width-1)/2)
            empty_slice = [0] * ybound
            for index in range(ylowr, yuppr+1):
                indx_slice = reshape(ODshot[:,index], ybound) #488
                empty_slice = empty_slice + indx_slice
            avgd_slice = empty_slice/pixel_width
            pixels = np.arange(0, ybound, 1) #488
        else:
            print_time('No averaged slice produced')
    return avgd_slice, pixels
    
def thomas_fermi_1d_fit(xx, x0, R_tf, amplitude, offset):
    ThomasFermi = amplitude * ((1.0-(xx-x0)**2/R_tf**2)) + offset
    ThomasFermi[ThomasFermi < 0] = 0
    return ThomasFermi

def gaussian_1d_fit(x, x0, sigma_x, amplitude, offset):
    gauss = amplitude * exp(-0.5*((x-x0)/sigma_x)**2) + offset
    return gauss
	
try:
    print_time('Get Raw shots from %s' % os.path.basename(path))

    if not image_group in 'data':
        print_time('Calculating OD...')
    
        OD = raw_to_OD('Raw') 
        ODrot = interpolation.rotate(OD, 3.8)
        ODresize = ODrot[0:OD.shape[0]:1, 0:OD.shape[1]:1]
        ROI, ROI_centroid, OD_ROIcenter = get_ROI_box(ODrot, OD, threshold_OD = -0.5)
        ROI_bound = ROI.shape
        colOD, pix = slice_width_avgd(ROI, 3, OD_ROIcenter, ROI_bound, xslice=True)
        ODslice = colOD.T
        x_slice = pix * pixel_size / 1e-6
        xlim = np.amax(x_slice)
   
        # PERFORM THOMAS-FERMI FIT AND GAUSSIAN FIT
    
	    # Gaussian 1D
        gauss_initial_guess = ([np.argmax(ODslice), np.argmax(ODslice), np.amax(ODslice), 0.1])
        gaussian_fit_params, gauss_cov = curve_fit(gaussian_1d_fit, x_slice, ODslice, p0 = gauss_initial_guess)
        print '    Center             Width          peak OD            Offset', '\n',  gaussian_fit_params
        gaussian_A = gaussian_1d_fit(x_slice, gaussian_fit_params[0], gaussian_fit_params[1], gaussian_fit_params[2], gaussian_fit_params[3])
        if gaussian_A is not ([]):
            run.save_result('gauss_width', gaussian_fit_params[1])
            print_time('Gauss fit successful for ROI')
        else:
            raise Exception('Can \'t fit Gauss profile')

        # Thomas Fermi 1D 
        tf_initial_guess = ([np.argmax(ODslice), gaussian_fit_params[1]/2, np.amax(ODslice), 0.0])
        tf_fit_params, tf_cov = curve_fit(thomas_fermi_1d_fit, x_slice, ODslice, p0 = tf_initial_guess)
        print '    Center         Width       Amplitude      Offset', '\n', tf_fit_params
        thomas_fermi_A = thomas_fermi_1d_fit(x_slice, tf_fit_params[0], tf_fit_params[1], tf_fit_params[2], tf_fit_params[3])
        tf_null = thomas_fermi_A[thomas_fermi_A == 0.0].shape[0]
        tf_full = thomas_fermi_A.shape[0]
        rTF = int((tf_full - tf_null)/2)
        if thomas_fermi_A is not ([]):
            run.save_result('R_tf', rTF)
            print_time('TF fit successful for ROI')
        else:
             raise Exception('Can \'t fit TF profile')

        n_1d = ODslice*pixel_size/sigma0
        n_1dfit= gaussian_A*pixel_size/sigma0
        gamma = 11.6e-9*1.44e-25*27e3/(1.05e-34*n_1d)
        gamma_fit = 11.6e-9*1.44e-25*27e3/(1.05e-34*n_1dfit)
        
        #plot(x_slice, gamma, 'b.', x_slice, gamma_fit, 'r')
        plot(x_slice, ODslice, 'b', x_slice, thomas_fermi_A, 'r', x_slice, gaussian_A, 'g', label='OD')
        #axis([0, xlim, -0.0, 5])
        xlabel('z (um)')
        ylabel('OD')
        title('Slice OD')
        show()
        print_time('Plot OD and slice...')
               
        # Total atom number
        N = colOD.sum()*3*3*pixel_size**2/sigma0
        print N
        fig = figure()
        imshow(ROI, vmin= -0.4, vmax = 2.0, cmap = 'nipy_spectral')
        colorbar()
        draw()
        run.save_result('N', N)
        #run.save_result('ytODslice', ODslice)
    else:
        print_time('Unsuccessful...')
        raise Exception( 'No image found in file...' )

except Exception as e:
    print str(e) + ' ' +  os.path.basename(path)
    print '\n ********** Not Successful **********\n\n'