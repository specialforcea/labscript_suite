# Takes RAW arrays and returns calculated OD for given shot
# along with the best fit (between gaussian and TF) for ROI.

from __future__ import division
from time import time
import h5py
import os as os
import numpy as np

# For LYSE
from lyse import *


from pylab import *
import matplotlib.pyplot as plt
import lmfit
import PhysConstants as phys
import QgasFunctionsLmfit as Qfunc
import QgasUtils as Util
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Folder and file locations
dataFolder = "//129.6.128.236/Labscript_Shared/Experiments/rb_chip/"
sequenceFile = "BEC_BraggHunting"
year = "2016"
month = "05"
day = "04"
seqIndex = "0030"
idStart = 00
idEnd = 25

# TODO: consider using the os module to get filename from the folder.
foldername = dataFolder + sequenceFile + "/" + year + "/" + month + "/" + day + "/" +seqIndex + "/"
fileheader = year + "_" + month + "_" + day + "_" + sequenceFile + "_"
filename = foldername + fileheader + str("%.2d" % idStart) + ".h5"

## For LYSE
run = Run(path)
filename = path


# Parameters
pixel_size = 5.6e-6/3.15  # Divided by Magnification Factor

# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()

def print_time(s):
    print 't = %6.3f : %s' % ((time()-t), s)

print_time('Start running...')

# TODO: Right now this only works for absorption imaging. Should include imaging option in the program
def getOD(camID):
    img_abs = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[0])
    img_pro = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[1])
    img_bkg = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[2])
    img_OD = np.flipud(-np.log((img_abs-img_bkg)/(img_pro-img_bkg)))
    
    return img_OD
    
def getODcorr(camID):
    Isat = 297;
    alpha = 1.645;
    img_abs = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[0])
    img_pro = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[1])
    img_bkg = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[2])
    return np.flipud(-alpha*np.log((img_abs-img_bkg)/(img_pro-img_bkg)) - ((img_abs-img_pro)/Isat))
    

def getODfluor(camID):
    img_abs = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[0])
    img_pro = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[1])
    #img_bkg = np.ma.masked_invalid(array(h5_file['data'][camID]['Raw'])[2])
    return np.flipud((img_abs-img_pro))
    
def imshowImg(Image,zmin,zmax,axis,showcbar = False):
    Z = np.flipud(Image['OptDepth'])
    xmin = Image['xVals'].min()
    ymin = Image['yVals'].min()
    xmax = Image['xVals'].max()
    ymax = Image['yVals'].max()
    im = axis.imshow(Z, cmap=plt.get_cmap('jet'), vmin=zmin, vmax=zmax, extent=[xmin, xmax, ymin, ymax], aspect = 'auto', interpolation = 'none')
    
    if showcbar:
        divider = make_axes_locatable(plt.gca())
        cbsize = divider.append_axes("right", "5%", pad="-6%") 
        cb = plt.colorbar(im, cax=cbsize)
    #cb.yaxis.set_ticks_position('right')
    #cb.yaxis.set_label_position('right')
    #im.set_interpolation('bilinear')
        
# Open h5 file, load whatever information needed, and close file
with h5py.File(filename) as h5_file:
    insitu_only = h5_file['globals/imaging'].attrs['insitu_only']
    if insitu_only == 'True':
        camID = 'imagesYZ_1_Flea3'
	  #camID = 'imagesXY_2_Flea3'
    else:
        camID = 'imagesXY_1_Flea3'
    Image = {}
    Image["OptDepth"] = getODfluor(camID)

# TODO: this declares the axis scaling. We should implement the right factor from magnification and pixel size    
xarray = np.linspace(0.5,648,Image["OptDepth"].shape[1]);
yarray = np.linspace(0.5,488,Image["OptDepth"].shape[0]);

(xVals,yVals) = np.meshgrid(xarray,yarray);
Image["xVals"] = xVals;
Image["yVals"] = yVals;

print_time('Test plotting...')
#createPlots()
# MAIN IMAGE SHOW
plt.figure(1, figsize=(6,6))
gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1])
ax2 = plt.subplot(gs[0])
imshowImg(Image,Image['OptDepth'].min(),Image['OptDepth'].max(),ax2, True)
img = Util.ImageSliceFromMax(Image, 5, pScale = True)

pguess = lmfit.Parameters()
pguess.add('offset', value = 0.5)
pguess.add('A', value = 3)
pguess.add('x0', value = img['xMax'])
pguess.add('R', value = 50)

ax4 = plt.subplot(gs[2], sharex=ax2)
ax4.plot(img['xVals'], img['xSlice'])
fit_par = Qfunc.FitQgas(img['xVals'], img['xSlice'], Qfunc.TF_only, pguess)
fitdata2 = Qfunc.TF_only(img['xVals'], **fit_par.values)
ax4.plot(img['xVals'],fitdata2)

peakOD = fit_par.params['A'].value/2.0;
Rxfit =  fit_par.params['R'].value;
pguess.add('x0', value = img['yMax'])
pguess.add('xc', value = 0)

ax1 = plt.subplot(gs[1], sharey=ax2)
ax1.set_xlim(ax4.get_ylim())
ax1.plot(img['ySlice'], img['yVals'])
fit_par = Qfunc.FitQgas(img['yVals'], img['ySlice'], Qfunc.TF_onlyChopped, pguess)
fitdata = Qfunc.TF_onlyChopped(img['yVals'], **fit_par.values)
ax1.plot(fitdata, img['yVals'])
ax1.yaxis.set_ticks_position('right')
ax1.yaxis.set_label_position('right')

plt.tight_layout()

peakOD = fit_par.params['A'].value/2.0 + peakOD;
Ryfit =  fit_par.params['R'].value;


print_time('Do cropping...')
# CROPPED IMAGE SHOW
p1c = np.array([160,240])
z0c = np.array([320,245])
m1c = np.array([485,250])
cropsize = np.array([100,100])

bkc = np.array([150,400])
CroppedBkg = Util.ImageCrop(Image['xVals'], Image['yVals'], Image['OptDepth'], bkc, cropsize, Scaled = True, Center = False)
Image['OptDepth'] = Image['OptDepth'] - np.mean(CroppedBkg['OptDepth'])


Z = Image['OptDepth']
zmin = Z.min()
zmax = Z.max()

plt.figure(2, figsize=(6,6))
gs = gridspec.GridSpec(4, 1)



ax1 = plt.subplot(gs[2])
Cropped1 = Util.ImageCrop(Image['xVals'], Image['yVals'], Image['OptDepth'], p1c, cropsize, Scaled = True, Center = False)
imshowImg(Cropped1, zmin, zmax, ax1)

imgSlice = Util.ImageSliceFromMax(Cropped1, 3, pScale = True)

pguess = lmfit.Parameters()
pguess.add('offset', value = 1)
pguess.add('A', value = 3)
pguess.add('x0', value = imgSlice['xMax'])
pguess.add('R', value = 50)
pguess.add('xc', value = 0)



ax4 = plt.subplot(gs[3], sharex=ax1)
fit_par = Qfunc.FitQgas(imgSlice['xVals'], imgSlice['xSlice'], Qfunc.TF_onlyChopped, pguess)

pguess.add('x0', value = imgSlice['yMax'])
fit_pary = Qfunc.FitQgas(imgSlice['yVals'], imgSlice['ySlice'], Qfunc.TF_onlyChopped, pguess)
pOneX = fit_par.params['x0'].value
pOneY = fit_pary.params['x0'].value

fitdata2 = Qfunc.TF_onlyChopped(imgSlice['xVals'], **fit_par.values)
ax4.plot(imgSlice['xVals'], fitdata2)



ax2 = plt.subplot(gs[1], sharex=ax1)
Cropped2 = Util.ImageCrop(Image['xVals'], Image['yVals'], Image['OptDepth'], z0c, cropsize, Scaled = True, Center = False)
imshowImg(Cropped2, zmin, zmax, ax2)

imgSlice = Util.ImageSliceFromMax(Cropped2, 3, pScale = True)

pguess.add('x0', value = imgSlice['xMax'])
fit_par = Qfunc.FitQgas(imgSlice['xVals'], imgSlice['xSlice'], Qfunc.TF_onlyChopped, pguess)
pguess.add('x0', value = imgSlice['yMax'])
fit_pary = Qfunc.FitQgas(imgSlice['yVals'], imgSlice['ySlice'], Qfunc.TF_onlyChopped, pguess)

zeroX = fit_par.params['x0'].value
zeroY = fit_pary.params['x0'].value

fitdata2 = Qfunc.TF_onlyChopped(imgSlice['xVals'], **fit_par.values)
ax4.plot(imgSlice['xVals'], fitdata2)



ax3 = plt.subplot(gs[0], sharex=ax1)
Cropped3 = Util.ImageCrop(Image['xVals'], Image['yVals'], Image['OptDepth'], m1c, cropsize, Scaled = True, Center = False)
imshowImg(Cropped3, zmin, zmax, ax3)

imgSlice = Util.ImageSliceFromMax(Cropped3, 3, pScale = True)

pguess.add('x0', value = imgSlice['xMax'])
fit_par = Qfunc.FitQgas(imgSlice['xVals'], imgSlice['xSlice'], Qfunc.TF_onlyChopped, pguess)
pguess.add('x0', value = imgSlice['yMax'])
fit_pary = Qfunc.FitQgas(imgSlice['yVals'], imgSlice['ySlice'], Qfunc.TF_onlyChopped, pguess)

mOneX = fit_par.params['x0'].value
mOneY = fit_pary.params['x0'].value

fitdata2 = Qfunc.TF_onlyChopped(imgSlice['xVals'], **fit_par.values)
ax4.plot(imgSlice['xVals'], fitdata2)


plt.tight_layout()


print_time('Save stuff to be analyzed...')

scalefac = 5.6/3.15;
pOne = phys.NumberCalcAbsOD(Cropped1['OptDepth'],scalefac)
zero = phys.NumberCalcAbsOD(Cropped2['OptDepth'],scalefac)
mOne = phys.NumberCalcAbsOD(Cropped3['OptDepth'],scalefac)

print pOne,
print zero,
print mOne


run.save_result('pOne', pOne)
run.save_result('zero', zero)
run.save_result('mOne', mOne)

run.save_result('pOneX', pOneX)
run.save_result('zeroX', zeroX)
run.save_result('mOneX', mOneX)

run.save_result('pOneY', pOneY)
run.save_result('zeroY', zeroY)
run.save_result('mOneY', mOneY)

run.save_result('Rxfit', Rxfit)
run.save_result('Ryfit', Ryfit)
run.save_result('peakOD', peakOD)
#xx = np.linspace(0,100,101);
#yy = Qfunc.stepfn(xx, 0, 1, 50, 10);
#plot(xx,yy)


