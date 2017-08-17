from __future__ import division
from lyse import *
from time import time
from matplotlib import pyplot as plt
from common.OD_handler import ODShot
from analysislib.spinor.aliases import *
import os
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle

""" plot_imb_gauge based on the matplotlib gauge drawing from
    http://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/ 
    which I found really awesome"""

# Parameters
pixel_size = 5.6e-6/5.33# Divided by Magnification Factor
            # 5.6e-6/5.33 for z in situ                        # Yuchen and Paco: 08/19/2016
            #5.6e-6/3.44 for z TOF                           # Yuchen and Paco: 08/19/2016
            #5.6e-6/2.72 for x-in situ                        # Paco: 05/06/2016
            
# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()

# Load dataframe
run = Run(path)

# Methods
def print_time(text):
    print 't = %6.3f : %s' % ((time()-t), text)

def u_wave_raw_to_OD(fpath):
    with h5py.File(fpath) as h5_file:
        # Safe to assume uwave lock is *only* along z-insitu
        if  '/data/feed_forward' in h5_file: 
            Nimb = h5_file['/data/feed_forward']['ODs'].attrs['Nimb']
            return Nimb, h5_file['/data/feed_forward']['ODs'][0],  h5_file['/data/feed_forward']['ODs'][1]

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation
    
def plot_imb_gauge(OD1, OD2, labels=['-1.0','-0.5','0.0','0.5','1.0'], \
          colors='jet_r', arrow=0, title='', fname=False):  
    """ some sanity checks first """
    N = len(labels)
    if np.int(arrow) > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))
    """if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors """
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))
    """ begins the plotting """
    fig = plt.figure(figsize=(8, 5), frameon=False)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[1,1])
    ax0 = plt.subplot(gs[0])
    ax0.imshow(OD1, vmin=0., vmax=0.2, cmap='Reds', aspect='auto', interpolation='none')
    ax1 = plt.subplot(gs[1])
    ax1.imshow(OD2, vmin=0., vmax=0.2, cmap='Blues', aspect='auto', interpolation='none')
    ax2 = plt.subplot(gs[2])
    ang_range, mid_points = degree_range(N)
    labels = labels[::-1]
    """plots the sectors and the arcs """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    [ax2.add_patch(p) for p in patches]
    """set the labels"""
    for mid, lab in zip(mid_points, labels): 
        ax2.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=14, \
            rotation = rot_text(mid))
    """set the bottom banner and the title"""
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax2.add_patch(r) 
    ax2.text(0, -0.05, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=16)
    """ plots the arrow """
    pos = (1-arrow)*90 #mid_points[abs(arrow - N)]
    ax2.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                    width=0.02, head_width=0.06, head_length=0.1, fc='k', ec='k')
    ax2.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax2.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))
    """ removes frame and ticks, and makes axis equal and tight """
    ax2.set_frame_on(False)
    ax2.axes.set_xticks([])
    ax2.axes.set_yticks([])
    ax2.axis('equal')
    plt.tight_layout()
    plt.show()
        
        
# Main
try:
    with h5py.File(path) as h5_file:
        if '/data' in h5_file:
        # Get OD
            N_imb, ODa, ODb = u_wave_raw_to_OD(path)
            run.save_result('Imbalance', N_imb)
            plot_imb_gauge(ODa, ODb, labels=['-1.0','-0.5','0.0','0.5','1.0'], colors='RdBu', arrow=N_imb, title='Imbalance', fname=False)
except Exception as e:
    print '%s' %e +  os.path.basename(path)
    print '\n ********** Not Successful **********\n\n'
            
            
            
            

            
