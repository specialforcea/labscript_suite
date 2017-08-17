# Takes RAW arrays and returns calculated OD for given shot
# along with the best fit (between gaussian and TF) for ROI.

from __future__ import division
from lyse import *
from pylab import *
from analysislib.common.fit_gaussian_2d import fit_2d
from analysislib.common.traces import *
from analysislib.spinor.aliases import *
from time import time
from scipy.ndimage import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import labscript_utils.h5_lock, h5py
import os
import pandas as pd
import numpy as np
import numexpr as ne
import matplotlib.gridspec as gridspec
from analysislib import fit_table


# Parameters
pixel_size = 5.6e-6/3.15  # Divided by Magnification Factor

# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()
def print_time(s):
    print 't = %6.3f : %s' % ((time()-t), s)
	
# Run dataframe to save result and do multishot analysis
run = Run(path)

with h5py.File(path) as h5_file:
    insitu_only = h5_file['globals/imaging'].attrs['insitu_only']
    # print insitu_only
    if insitu_only == 'True':
        image_group = 'imagesXY_1_Flea3'
        #image_group = 'imagesYZ_1_Flea3'
    if not insitu_only:
        #image_group = 'imagesYZ_1_Flea3'
        image_group = 'imagesXY_1_Flea3'

# Methods and Functions

def raw_to_OD(name):
     with h5py.File(path) as h5_file:
        abs = ma.masked_invalid(array(h5_file['data'][image_group][name])[0])
        probe = ma.masked_invalid(array(h5_file['data'][image_group][name])[1])
        bckg = ma.masked_invalid(array(h5_file['data'][image_group][name])[2])
        print_time('Get Raw shots from %s' % os.path.basename(path))
        M = matrix(-log((abs - bckg)/(probe - bckg)))
        return M
   
def get_ROI_guess(multiple_ROI):
    with h5py.File(path) as h5_file:
        multi_ROI = True #h5_file['globals/lyse_wise'].attrs['multiple_ROI']
        n_ROIs = 2 #h5_file['globals/lyse_wise'].attrs['n_ROIs']
        if multi_ROI:
            x0 = 1  # In development
            xf = 487  # In development
            y0 = 1  # In development
            yf = 647  # In development
            x0_b = 0  # In development
            xf_b = 488  # In development
            y0_b = 0  # In development
            yf_b = 648  # In development
            return x0, xf, y0, yf, x0_b, xf_b, y0_b, yf_b
        else:
            #x0 = int(h5_file['globals/lyse_wise'].attrs['ROI_box_x0'])
            #xf = int(h5_file['globals/lyse_wise'].attrs['ROI_box_xf'])
            #y0 = int(h5_file['globals/lyse_wise'].attrs['ROI_box_y0'])
            #yf = int(h5_file['globals/lyse_wise'].attrs['ROI_box_yf'])
            #x0_b = int(h5_file['globals/lyse_wise'].attrs['x0_background'])
            #xf_b = int(h5_file['globals/lyse_wise'].attrs['xf_background'])
            #y0_b = int(h5_file['globals/lyse_wise'].attrs['y0_background'])
            #yf_b = int(h5_file['globals/lyse_wise'].attrs['yf_background'])
            x0 = 250; xf = 400; x0_b = x0-10; xf_b = xf+10;
            y0 = 280; yf = 470; y0_b = y0-10; yf_b = yf+10;
            

        # Check if background box encloses ROI box
            print 'ROI is happening'
            if (y0-y0_b>0 and x0-x0_b>0) and (yf-yf_b<0 and xf-xf_b<0):
                return x0, xf, y0, yf, x0_b, xf_b, y0_b, yf_b
            else:
                raise Exception( 'Background does not include ROI... ')
                print x0, xf, y0, yf, 0,0,0,0
        
def get_ROI_box(ODshot, ODuntrans, threshold_OD):
    
    x_off = ODshot.shape[0] - ODuntrans.shape[0]
    y_off = ODshot.shape[1] - ODuntrans.shape[1]

    artif_backg = np.zeros(ODshot.shape)
    OD_thrs = ma.masked_where(ODshot < threshold_OD, ODshot) + artif_backg  

    x0, xf, y0, yf, x0_b, xf_b, y0_b, yf_b = get_ROI_guess(True)
    print x0, y0, xf, yf
    ROI_box = OD_thrs[x0:xf+x_off:1, y0:yf+y_off:1]
    ROI_COM = measurements.center_of_mass(ROI_box)

    max_guess = measurements.maximum_position(array(ROI_box))
    xx = int(max_guess[0])
    yy = int(max_guess[1])
    
    ROI_abs_center = ([xx, yy])#([x0 + (xf-x0)/2, y0 + (yf-y0)/2])

    return ROI_box, ROI_COM, ROI_abs_center        


    
def get_region(ODshot, x1, x2, y1, y2):
	ROI_region = ODshot[x1:x2:1, y1:y2:1]
	
	return ROI_region
	
	
def slice_width_avgd(ODshot, pixel_width, center, bound):
    if (pixel_width % 2 == 0):
        raise Exception('The argument pixel width'
		                'should be odd so that it'
                        'covers an evenly split range')
    else:
        xpos = center[0]
        ypos = center[1]
        xbound = bound[1]
        ybound = bound[0]
                
        xuppr = int(xpos + (pixel_width-1)/2)
        xlowr = int(xpos - (pixel_width-1)/2)
        x_empty_slice = [0] * xbound #648
        for index in range(xlowr, xuppr+1):
            x_indx_slice = reshape(ODshot[index,:], xbound) #648
            x_empty_slice = x_empty_slice + x_indx_slice
        x_avgd_slice = x_empty_slice/pixel_width
        x_pixels = np.arange(0, xbound, 1) #648
        x_axis = x_pixels * pixel_size / 1e-6 # In um

        yuppr = int(ypos + (pixel_width-1)/2)
        ylowr = int(ypos - (pixel_width-1)/2)
        y_empty_slice = [0] * ybound
        for index in range(ylowr, yuppr+1):
            y_indx_slice = reshape(ODshot[:,index], ybound) #488
            y_empty_slice = y_empty_slice + y_indx_slice
        y_avgd_slice = y_empty_slice/pixel_width
        y_pixels = np.arange(0, ybound, 1) #488
        y_axis = y_pixels * pixel_size / 1e-6 # In um
        
    return x_avgd_slice.T, y_avgd_slice.T, x_axis, y_axis

	
def get_background_level(ODshot, threshold_OD):
    
    artif_backg = np.zeros(ODshot.shape)
    OD_thrs = ma.masked_where(ODshot < threshold_OD, ODshot) + artif_backg  
    
    x0, xf, y0, yf, x0_b, xf_b, y0_b, yf_b = get_ROI_guess(True)
    
    ROI_box = OD_thrs[x0:xf:1, y0:yf:1]
    background_box = OD_thrs[x0_b:xf_b:1, y0_b:yf_b:1]
    
    background_box[x0-x0_b:xf-x0_b:1, y0-y0_b:yf-y0_b:1] = np.zeros(background_box[x0-x0_b:xf-x0_b:1, y0-y0_b:yf-y0_b:1].shape)  
    area_avg = np.count_nonzero(background_box)

    avg_bck = background_box.sum()/area_avg
    BCK = np.ones(ROI_box.shape) * avg_bck
    
    #imshow((background_box), cmap = 'jet', aspect='auto')
    return BCK

	
# Main
try:
    if not image_group in 'data':
        print_time('Calculating OD...')
    # Get OD from ROI coordinates and plot it
        OD = raw_to_OD('Raw')
        ODrot = interpolation.rotate(OD, 0.0)
        ODresize = ODrot[0:OD.shape[0]:1, 0:OD.shape[1]:1]
        ROI, ROI_centroid, OD_ROIcenter = get_ROI_box(ODrot, OD, threshold_OD = -0.5)
        ROI_bound = ROI.shape
        ROItilde = np.fft.fft2(ROI)
        ROIfft = np.fft.fftshift(ROItilde)
        ROIlow = filters.gaussian_filter(ROI, 2)
        ROIfilter = ROI - ROIlow
    # Calculate number of atoms from total OD
        BCK = get_background_level(OD, threshold_OD = -0.5)
        #N = (ROI-BCK).sum()*pixel_size**2/sigma0
        N = (ROI).sum()*pixel_size**2/sigma0;
		
        ROI1 = get_region(ODrot,150,240,440,530);
        
        #ROI2 = get_region(ODrot,110,200,420,520);
        #ROI2 = get_region(ODrot,250,340,280,390);
        ROI2 = get_region(ODrot,220,340,260,400);
        
        #ROI3 = get_region(ODrot,35,100,430,510);
        #ROI3 = get_region(ODrot,340,430,100,220);
        ROI3 = get_region(ODrot,340,460,260,400);
        
        N1 = (ROI1).sum()*pixel_size**2/sigma0;
        N2 = (ROI2).sum()*pixel_size**2/sigma0;
        N3 = (ROI3).sum()*pixel_size**2/sigma0;
        N1 = 0;
        
        Nfrac1 = (N1)/(N1+N2+N3);
        Nfrac2 = (N2)/(N1+N2+N3);
        Nfrac3 = (N3)/(N1+N2+N3);
        Ntotal = N1+N2+N3;
        print N #
        run.save_result('N', N)
        run.save_result('Nfrac1', Nfrac1)
        run.save_result('Nfrac2', Nfrac2)
        run.save_result('Nfrac3', Nfrac3)
        run.save_result('Ntotal', Ntotal)
        run.save_result('N3', N3)
		
    # Display figure with shot, slices and fits and N
        fig2 = figure()
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,1])
        subplot(gs[0])
        im01= imshow((ROI1), vmin= -0.0, vmax =  3.0, cmap = 'jet', aspect='auto', interpolation='none')
        subplot(gs[1])
        im02= imshow((ROI2), vmin= -0.0, vmax =  3.0, cmap = 'jet', aspect='auto', interpolation='none')
        subplot(gs[2])
        im02= imshow((ROI3), vmin= -0.0, vmax =  3.0, cmap = 'jet', aspect='auto', interpolation='none')
		
		
		
        fig = figure()        
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[4,1]) 
        subplot(gs[2])
        str = r'N = %.0f' % N
        text(0.4, 0.6, str, ha='center', va='top',fontsize=18)
        gca().axison = False
        tight_layout()
    # OD shot          
        subplot(gs[1])
        im0= imshow((ROI), vmin= -0.0, vmax =  3.0, cmap = 'jet', aspect='auto', interpolation='none')
        divider = make_axes_locatable(gca())
        cax = divider.append_axes("right", "5%", pad="3%") 
        colorbar(im0, cax=cax)
        title('OD')
        tight_layout()
    # Slice routine
        print_time('Slice and fit...')
        xcolOD, ycolOD, x_ax, y_ax = slice_width_avgd(ROI, 3, OD_ROIcenter, ROI_bound)
    # Save center of cloud
        run.save_result('xROIcenter', OD_ROIcenter[0])
        run.save_result('yROIcenter', OD_ROIcenter[1])
    # Raw data is displayed and if fits are unsuccesful only show raws.
        # Gaussian 1D
        x_gaussian_par, dx_gaussian_par = fit_gaussian(x_ax, xcolOD)
        y_gaussian_par, dy_gaussian_par = fit_gaussian(y_ax, ycolOD)
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
            #fit_table.get_params(dy_tf_par)
            x_tf_fit = thomas_fermi(x_ax, x_tf_par[0], x_tf_par[1], x_tf_par[2], x_tf_par[3])
            y_tf_fit = thomas_fermi(y_ax, y_tf_par[0], y_tf_par[1], y_tf_par[2], y_tf_par[3])
            if(x_tf_fit is not None or y_tf_fit is not None): 
                print_time('TF fit successful for x and y')
                subplot(gs[3])
                plot(x_ax, xcolOD, 'b', x_ax, x_gaussian_fit, 'r', x_ax, x_tf_fit, 'g')
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

except Exception as e:
    print '%s' %e +  os.path.basename(path) 
    print '\n ********** Not Successful **********\n\n'