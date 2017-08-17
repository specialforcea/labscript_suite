from __future__ import division
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
import matplotlib.pyplot as plt
from scipy.linalg import pinv, lu, solve
from sklearn.linear_model import Lasso

plt.clf()
__author__ = 'dimitris'

def fringeremoval(img_list, ref_list, mask='all', method='svd'):

    nimgs = len(img_list)
    nimgsR = len(ref_list)
    xdim = img_list[0].shape[0]
    ydim = img_list[0].shape[1]

    if mask == 'all':
        bgmask = np.ones([ydim, xdim])
        # around 2% OD reduction with no mask
    else:
        bgmask = mask
        
    k = (bgmask == 1).flatten(1)
    
    # needs to be >float32 since float16 doesn't work with linalg
    
    R = np.dstack(ref_list).reshape((xdim*ydim, nimgsR)).astype(np.float32)
    A = np.dstack(img_list).reshape((xdim*ydim, nimgs)).astype(np.float32)
     
    # Timings: for 50 ref images lasso is twice as slow
    # lasso 1.00
    # svd 0.54
    # lu 0.54
    
    optref_list = []
    
    for j in range(A.shape[1]):
        
        if method == 'svd':
            b = R[k, :].T.dot(A[k, j])
            Binv = pinv(R[k, :].T.dot(R[k, :])) # svd through pinv
            c = Binv.dot(b)
            # can also try linalg.svd()
            
        elif method == 'lu':
            b = R[k, :].T.dot(A[k, j])
            p, L, U = lu(R[k, :].T.dot(R[k, :]))
            c = solve(U, solve(L, p.T.dot(b)))
            
        elif method == 'lasso':
            lasso = Lasso(alpha=0.01)
            lasso.fit(R, A)
            c = lasso.coef_
            
        else:
            raise Exception('Invalid method.')
        
        optref_list.append(np.reshape(R.dot(c), (xdim, ydim)))
    
    return optref_list
    
# Shots filenames
ref_filepath = r'\\129.6.128.236\Labscript_Shared\Experiments\rb_chip\1D_parametric_heat\2016\10\17\0003'
ref_files = [f for f in listdir(ref_filepath) if isfile(join(ref_filepath, f))]
del ref_files[-1]  # Gets rid of notes.txt, usually at the end

shot_filepath = r'\\129.6.128.236\Labscript_Shared\Experiments\rb_chip\1D_parametric_heat\2016\10\17\0003'
shotfiles = [f for f in listdir(shot_filepath) if isfile(join(shot_filepath, f))]
del shotfiles[-1]

probe_list = []
shots = []
pkOD = []
# little improvement on the fit when using >20 probe pictures
# create a list of all probe pics
for each in ref_files:
    with h5py.File(join(ref_filepath, each)) as h5_file:
        probe_list.append(np.array(h5_file['data']['imagesXY_2_Flea3']['Raw'])[1][:,:])

# load an image to fit to
def raw_to_OD(filep, fpath):
    with h5py.File(join(filep, fpath)) as h5_file:
        insitu_only = h5_file['globals/imaging'].attrs['insitu_only']
        if insitu_only:
            image_group = "imagesXY_2_Flea3"
            Isat = 395      # In counts // Paco:04/20/2016
            alpha= 1#4.3
        elif not insitu_only:
            image_group = 'imagesXY_1_Flea3'
            Isat = 297          # In counts // Paco:04/21/2016
            alpha = 1.645
        atoms = np.array(h5_file['data'][image_group]['Raw'])[0] 
        probe = np.array(h5_file['data'][image_group]['Raw'])[1]
        bckg = np.array(h5_file['data'][image_group]['Raw'])[2]
        div = ((atoms - bckg)/(probe - bckg))
        div = np.ma.masked_invalid(div)
        div = np.ma.masked_less_equal(div, 0.)
        #print 'Got Raw shots from %s' % os.path.basename(path)
        another_term = 0*(probe-atoms)/(Isat)
        #saved_number = h5_file['results/get_OD'].attrs['N_(1,-1)']
    return np.matrix(-alpha*np.log(div)+another_term)
    
for pic in shotfiles:    
    shot = [np.array((raw_to_OD(shot_filepath, pic)))]
    # call the function and plot the output
    optref = fringeremoval(shot, probe_list, method='lasso')
    opt_shot = np.log(((shot[0] < 1) + shot[0]) / ((optref[0] < 1) + optref[0]))
    pkOD.append(np.amax(opt_shot))
# compare with od calculated in qgasfileio
#plt.imshow(shot[0], vmin=-0.0, vmax=0.5, interpolation='None')
#plt.figure()
#plt.imshow(opt_shot,vmin=-0.0, vmax=0.5, interpolation='None')