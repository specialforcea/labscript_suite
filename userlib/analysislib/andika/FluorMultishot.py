from __future__ import division
#from lyse import *
import os
from time import time
import matplotlib.pyplot as plt
import QgasFunctionsLmfit as Qfunc
import matplotlib.gridspec as gridspec
import h5py
from scipy.stats import binned_statistic

#import pandas as pd
import numpy as np
import lmfit


dataFolder = "//129.6.128.236/Labscript_Shared/Experiments/rb_chip/"
sequenceFile = "BEC_BraggHunting"
year = "2016"
month = "05"
day = "09"

#seqId = np.arange(43,85, dtype=np.float)
#seqId = np.array([40, 43, 49, 51, 53, 55, 58, 61, 63, 66, 70, 72, 74, 78, 81, 83, 84])
#seqId = np.array([91, 93, 94, 96, 98, 101, 104, 107, 109, 111, 112, 114, 116])
#seqId = np.array([93, 94, 96, 98, 109, 111, 112, 114, 116])
seqId = np.array([128, 129])
#seqId = np.array([30, 32, 33, 36, 37, 38, 40, 42, 44, 46, 48, 51, 53, 56])
fileExtList =['.h5']
xavg = []; yavg =[]; ystd =[]; pavg =[]; pstd=[];
#yavg = np.zeros_like(seqId)
#ystd = np.zeros_like(seqId)


marker = [];
ycounts = [];
peakOD = [];

for ss in seqId:
    seqIndex = '%04d' % ss
    print ss
#idStart = 00
#idEnd = 25

# TODO: consider using the os module to get filename from the folder.
    foldername = dataFolder + sequenceFile + "/" + year + "/" + month + "/" + day + "/" +seqIndex + "/"
    ii = 0.0;
    if os.path.isdir(foldername):
        fileList = [os.path.join(foldername, f) for f in os.listdir(foldername)
                    if os.path.splitext(f)[1] in fileExtList]
#print fileList
    
       # marker = [];
       # ycounts = [];
       # peakOD = [];

        for filename in fileList:
            #print filename
            with h5py.File(filename) as h5_file:
                #if 'rep' not in filename:
                    ycounts.append(float(h5_file['results/FluorAnalysis'].attrs['zero']))
               #    peakOD.append(float(h5_file['results/FluorAnalysis'].attrs['peak_OD']))
                    #marker.append(float(h5_file['globals'].attrs['trans_marker'])*30/1000000/40*180/np.pi)
                    marker.append(float(h5_file['globals'].attrs['raman_power']))
                #except KeyError:
                 #   print 'Not all files are analysed.'
        
    xavg.append(np.mean(marker))
    yavg.append(np.mean(ycounts))
    ystd.append(np.std(ycounts))
 #   pavg.append(np.mean(peakOD))
 #   pstd.append(np.std(peakOD))
        
        #xavg[ii] = np.mean(marker);
        #yavg[ii] = np.mean(ycounts);
        #ystd[ii] = np.std(ycounts);
    ii = ii + 1; 

#idx = np.argsort(marker)
#xavg = 157+72+200-np.array(marker)[idx]-80
#yavg = np.array(ycounts)[idx]
#binid = np.arange(-80.,-11.,4.)
#bin_means, bin_edges, binnumber = binned_statistic(xavg, yavg, statistic='mean', bins=binid)

#bin_means2, bin_edges, binnumber = binned_statistic(xavg, yavg**2, statistic='mean', bins=binid)
#xbin = np.arange(-79.,-14.,4.)
#ystd = np.sqrt(bin_means2-bin_means**2)
'''
plt.figure(1, figsize=(5,3))
plt.errorbar(xavg,yavg,ystd, marker='o', color='b', linestyle ='none')
#plt.errorbar(xbin,bin_means,ystd,marker='x', linestyle='none')
plt.ylabel('Signal counts (arb.units)', fontsize=12)
plt.xlabel('Detuning wrt 1->0 transition', fontsize=12)
plt.tight_layout()

'''
#plt.figure(4)
fig, ax1 = plt.subplots(figsize=(5,3))
ax1.plot(marker,ycounts,'bo')
''' 
ax1.errorbar(xavg,yavg,ystd,marker='o',color='b',linestyle='none')
ax1.set_ylabel('Signal counts (arb.units)', fontsize=12, color='b')
ax1.set_xlabel(r"Angle scan ($\Delta\Theta_i$)", fontsize=12)
ax1.set_ylim([0,160])
ax1.set_xlim([-1.2,2.5])

ax2 = ax1.twinx()
#ax2.plot(marker,peakOD,'rx')
ax2.errorbar(xavg,pavg,pstd,marker='x',color='r',linestyle='none')
ax2.set_ylabel('Peak OD (atoms no)', fontsize=12, color='r')
#plt.xlim([787.2,792.9])
ax2.set_ylim([0,6])
ax2.set_xlim([-1.2,2.5])
plt.tight_layout()
'''
        #if insitu_only == 'True':
        #    camID = 'imagesYZ_1_Flea3'
        #    #camID = 'imagesXY_2_Flea3'
        #else:
        #    camID = 'imagesXY_1_Flea3'
        #Image = {}; Background = {};
        #Image["OptDepth"] = np.flipud(array(h5_file['data'][camID]['Raw'])[0])
        #Background["OptDepth"] = np.flipud(array(h5_file['data'][camID]['Raw'])[1])
    
'''
print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()


#pulsetime = np.transpose(df['TimeRamanPulse'].get_values())[0]*1000000;
pulsetime = np.transpose(df['short_TOF'].get_values())[0];
mixercmd = np.transpose(df['bragg_power'].get_values())[0];
markerloc = np.transpose(df['rotation_marker'].get_values())[0];
#TpzShimRF = np.transpose(df['TpzShimRF'].get_values())[0];

print markerloc
zero = df['FluorAnalysis', 'zero'].get_values()
'''
'''
Ndim = len(Nfrac1)
runNo = np.linspace(0, Ndim-1, Ndim)
plt.figure(3)
plt.plot(runNo,Nfrac1,'bo-')
'''
'''
plt.figure(4)
plt.plot(mixercmd,zero,linestyle = "None", marker = "o")
plt.ylabel('counts', fontsize=12)
plt.xlabel('Rotation marker', fontsize=12)
plt.tight_layout()

#plt.plot(pulsetime,Ryfit/peakOD**(0.2),linestyle = "None", marker = "o")
#plt.plot(pulsetime,peakOD,linestyle = "None", marker = "o")
'''