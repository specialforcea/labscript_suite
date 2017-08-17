from __future__ import division
from lyse import *
from pylab import *
import os
from analysislib.common.fit_gaussian_2d import fit_2d
from analysislib.spinor.aliases import *
from time import time
# from matplotlib.patches import Ellipse, Path, PathPatch
import pandas as pd

# Parameters
do_fit = True
display_OD = True
save_results = True
m_list = [-1,0, 1]         # Magnetic quantum number of each ROI in order
# m_list = [-1, 1]         # Magnetic quantum number of each ROI in order
# m_list = [-1]              # Magnetic quantum number of each ROI in order
# fit_function = 'Gaussian'
fit_function = 'ThomasFermi'
OD_clip = 4.0               # Clip OD array to this value when fitting
OD_plot = (-0.1, 3.0)       # Plot limits
bin_width = 200             # Geometric mean width of rebinned frame for fitting
nan_value = 0               # Value to use for N_fit if fit fails or u_N_fit > N_fit or isnan(u_N_fit)


threshold_atom_number = 1e6#5e5  # max single component BEC population. Used as quick hack to stop bogus TF fitting of zero atom no.

print '\nRunning %s' % os.path.basename(__file__)
t = time()

def print_time(s):
    # return None		# suppress this for now
    print 't = %6.3f : %s' % ((time()-t), s)

if not spinning_top:
    sequence = '20140430150251'
    run_number = 0
    df = data()
    try:
        subdf = df.sequences(sequence)
        path = subdf['filepath'][run_number]
    except:
        path = df['filepath'][-1]

# Import data
camera = 'side'
image_group = 'images/'+camera+'/absorption/'    


run = Run(path)

try:
    print_time('Getting OD from %s' % os.path.basename(path))
    image_attrs = run.get_image_attributes(camera)
    
    if not 'roi0' in image_attrs.keys():
        print_time('No ROIs found.')
        
        OD_exists = True
        hardcode_rois = True
        
    else:
        print_time('Got OD')
        OD_exists = True
    OD = run.get_image(camera, 'absorption', 'OD')
except Exception as e:
    print '\n ********** CAMERA MISSED IMAGE **********\n'
    print str(e) + ' ' +  os.path.basename(path)
    print '\n ********** CAMERA MISSED IMAGE **********\n\n'
    OD_exists = False
    
if OD_exists:
    # atoms, flat, dark = run.get_images(camera, 'absorption', 'atoms', 'flat', 'dark')
    ydim, xdim = OD.shape
    pixel_size = image_attrs['effective_pixel_size']
    if 'OffsetX' in image_attrs:
        x_offset = image_attrs['OffsetX']
        y_offset = image_attrs['OffsetY']
    else:
        x_offset = image_attrs['roi_left']
        y_offset = image_attrs['roi_top']
    roi_offset = array([x_offset, y_offset, x_offset, y_offset])
    rois = []
    i = 0
    roi_exists = True
    while roi_exists:
        try:
            roi = amap(lambda x: x, image_attrs['roi%i' % i]) - roi_offset
            rois.append(roi)
            i += 1
        except KeyError:
            roi_exists = False
    #hardcode top ROIS here.
    # rois = array([(1116, 533, 1202, 595), (1258, 531, 1342, 595), (1409, 532, 1482, 595)]) - roi_offset
    # rois = array([(1087, 489, 1224, 595), (1226, 489, 1359, 595), (1361, 488, 1530, 595)]) - roi_offset
    
    if len(m_list) == 1 and len(rois) == 0:
        rois = [[0,0,image_attrs["Width"]-1,image_attrs["Height"]-1]]
    
    if run_globals['dipole_freq_0_transfer'] >= run_globals['dipole_freq_0'] or True:
       rois = rois[:len(m_list)]
       
    else:
       print 'Taking final %i ROIs!!! Beware when running this in concert with roi_atoms_split.py' % len(m_list)
       rois = rois[-len(m_list):]
       
    if len(m_list) != len(rois):
       print 'WARNING: more spin components than ROIs. Using m in ', m_list[:len(rois)]
    
    OD_list = []
    N_list = []
    N_fit_list = []
    
    for i, roi in enumerate(rois):
        ODi = OD[roi[1]:roi[3], roi[0]:roi[2]]
        ODi = ma.masked_invalid(ODi)
        ydim, xdim = ODi.shape
        N = ODi.sum()*pixel_size**2/sigma0
        N_list.append(N)
        if do_fit:
            print_time('Fitting %s to ROI %i...' % (fit_function, i))
            try:
                binsize = max(1, round((ydim*xdim)**0.5/bin_width)) # Enforce binned geometric mean width
                params_dict, X_section, Y_section = fit_2d(ODi, fit_function, binsize, clip=OD_clip)
                print_time('%s fit successful for ROI %i' % (fit_function, i))
                
                # Convert integrated number to units of atoms
                params_dict['%s_Nint' % fit_function] = tuple(pixel_size**2/sigma0*array(params_dict['%s_Nint' % fit_function]))
                N_fit, u_N_fit = params_dict['%s_Nint' % fit_function]
                print N_fit
                if u_N_fit > 0.5*N_fit or isnan(u_N_fit) or N_fit > threshold_atom_number:
                    N_fit = nan_value
                    if isnan(u_N_fit) or True:
                        u_N_fit = nan_value + 500
                    print 'Atom number set to %s for ROI %i in %s' % (nan_value, i, os.path.basename(path))
                N_fit_list.append((N_fit, u_N_fit))
                fit_success = True
            except Exception, e:
                print_time('Fit failed: %s' % str(e))
                N_fit_list.append((nan_value, nan_value))
                fit_success = False
        if save_results:
            ds = image_group + 'OD%i' % i
            run.save_result_array('OD%i' % i, ma.filled(ODi, nan).astype(float32), group=image_group)
            run.save_result('Nint', N, group=ds)
            if fit_success:
                run.save_results_dict(params_dict, uncertainties=True, group=ds)                    
                fit_args = {'fit_function': fit_function, 'binsize': binsize, 'clip': OD_clip}
                run.save_results_dict(fit_args, group=ds) 
        if display_OD:
            figure('OD%i, m = %i' % (i, m_list[i]))
            # Plot the OD with no interpolation
            # matshow(ODi, vmin=OD_plot[0], vmax=OD_plot[1], origin='upper')
            # figimage(ODi, vmin=OD_plot[0], vmax=OD_plot[1], origin='upper')
            # show()
            imshow(ODi)
    # Total atom number
    from uncertainties import ufloat
    for i, N_i in enumerate(N_list):
        print 'ROI %i: N_int = %7i, N_fit = %7i +/- %7i' % (i, N_i, N_fit_list[i][0], N_fit_list[i][1])
    N_list = array(N_list).clip(0, threshold_atom_number)   # array([N0, N1, ...])
    N_fit_list = array([ufloat(*v) for v in N_fit_list])    # array([N0_fit +/- u_N0_fit, ... ])
    Nt = ufloat(N_list.sum(), 0)                            # Total atom number (sum of integrated ODi)
    Nt_fit = N_fit_list.sum()                               # Total atom number (sum of fitted numbers)
    if Nt:
        rho_list = N_list/Nt                                    # Fractional populations (based on integrated ODi)
    else:
        rho_list = zeros_like(N_list)*ufloat(nan, 0)
    
    # Dimensionless spin projection
    if len(m_list) > 1:
        m_list = array(m_list)
        if Nt.n:
            Fz = sum(N_list*m_list)/Nt
        else:
            Fz = ufloat(nan, 0)
        if Nt_fit.n:
            Fz_fit = sum(N_fit_list*m_list)/Nt_fit
            rho_fit_list = N_fit_list/Nt_fit
        else:
            Fz_fit = ufloat(nan, 0)
            rho_fit_list = nan*N_fit_list
        for i in range(len(m_list)):
            run.save_result('rho_%i' % i, rho_list[i].n)
            run.save_result('rho_fit_%i' % i, rho_fit_list[i].n)
            run.save_result('u_rho_fit_%i' % i, rho_fit_list[i].s)
        
    # Save composite results
    from analysislib.common.utils import udictify
    if len(m_list) > 1:
        results_dict = udictify(Nt, Nt_fit, Fz, Fz_fit)
    else:
        results_dict = udictify(Nt, Nt_fit)
    results_dict = {key: (val.n, val.s) for key, val in results_dict.items()}
    run.save_results_dict(results_dict, uncertainties=True)
    
    # show()
    # figure('fitted atom number', figsize=(15,3), facecolor='w')
    # str = r'$N$ = %.3e' % Nt_fit.n
    # text(0.5,0.7,str,ha='center', va='top',fontsize=130)
    # gca().axison = False
    # show()
