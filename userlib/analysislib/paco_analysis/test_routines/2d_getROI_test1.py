
from __future__ import division
from lyse import *
from pylab import *
from time import time
from scipy.ndimage import *
from scipy.optimize import *
from matplotlib.widgets import Cursor, Button
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import numexpr as ne

plt.rcParams['font.size'] = 11

class cursors_ROI(object):
    def __init__(self, z, x = None, y = None):
        """
        Shows the calculated OD.
        Input : z; an OD 2d array.
        Output: ROI, BCK; 1d arrays.
        x, y coordinates are optional.
        """
        if x==None:
            self.x=np.arange(z.shape[0])
        else:
            self.x=x
        if y==None:
            self.y=np.arange(z.shape[1])
        else:
            self.y=y
        self.z=z

        global clickCounter
        clickCounter = 0
        self.ROIcoords = []
        self.BCKcoords = []

        self.fig = plt.figure()
        #Layout for subplots:
        self.fig.subplots_adjust(left = 0.08, bottom = 0.08, right = 0.93,
                                 top = 0.93, wspace = 0.50, hspace = 0.50)
        self.shot = plt.subplot2grid((9,5), (0,0), rowspan = 7, colspan = 3)
        self.shot.pcolormesh(self.z, cmap = 'RdBu_r')
        self.shot.autoscale(1, 'both', 1)
        self.ROI_subplot   = plt.subplot2grid((9,5), (0,3), rowspan = 3, colspan = 2)
        self.slice_subplot = plt.subplot2grid((9,5), (4,3), rowspan = 3, colspan = 2)

        #Cursor widget
        cursor = Cursor(self.shot, useblit = True, color = 'k', linewidth = 2)
        #Button widget
        but_ax  = plt.subplot2grid((9,5), (8,0), colspan = 1)
        reset_button   = Button(but_ax,  'Reset')
        but_ax2 = plt.subplot2grid((9,5), (8,1), colspan = 1)
        set_ROI_button = Button(but_ax2, 'Set ROI')
        but_ax3 = plt.subplot2grid((9,5), (8,2), colspan = 1)
        legend_button  = Button(but_ax3, 'Legend')
        # Widget List
        self._widgets = [cursor, reset_button, set_ROI_button, legend_button]

        #Connect events
        reset_button.on_clicked(self.clear_box)
        #set_ROI_button.on_clicked(self.get_coords)
        legend_button.on_clicked(self.get_coords)
        self.fig.canvas.mpl_connect('button_press_event', self.click)

    def show_subplots(self, event):
        """Shows subplots"""
        for pl in [self.ROI_subplot, self.slice_subplot]:
            if len(pl.lines) > 0:
                pl.legend()
        plt.draw()

    def clear_box(self, event):
        """Clears the ROI - BCK boxes"""
        self.ROIcoords  = []
        self.BCKcoords  = []
        self.shot.lines    = []
        self.ROI_subplot   = []
        self.slice_subplot = []
        global clickCounter
        clickCounter = 0
        plt.draw()

    def click(self, event):
        """
        What to do when a click on the figure happens:
            1. Set cursor
            2. Get coordinates
        """
        global clickCounter
        if clickCounter < 4:
            clickCounter += 1
            print clickCounter
            if event.inaxes == self.shot:
                xpos = np.argmin(np.abs(event.xdata - self.x))
                ypos = np.argmin(np.abs(event.ydata - self.y))
                if event.button   == 1:
                    #Plot ROI cursor
                    self.shot.axvline(self.x[xpos], color = 'black', lw = 1)
                    self.shot.axhline(self.y[ypos], color = 'black', lw = 1)
                    self.ROIcoords.append([xpos, ypos])
                elif event.button == 3:
                    #Plot BCK cursor
                    self.shot.axvline(self.x[xpos], color = 'b', lw = 1)
                    self.shot.axhline(self.y[ypos], color = 'b', lw = 1)
                    self.BCKcoords.append([xpos, ypos])
        else:
            print 'Done..'
        plt.draw()

    def counter_tracker(self):
        global clickCounter
        self._counter = clickCounter
        return self._counter

    def get_coords(self):
        print 'Getting coordinates...'
        return self.BCKcoords, self.ROIcoords

# Time stamp
print '\nRunning %s' % os.path.basename(__file__)
t = time()

if not spinning_top:
# if the script isn't being run from the lyse GUI,
# load the DataFrame from lyse and use the first shot
    try:
        df = data()
        path = df['filepath'][0]
    except:
        # if lyse isn't running, use an explicitly specified shot file path
        try:
            path = sys.argv[1]
        except:
            path = '20150325T151931_test_BEC_0_rep00028.h5'

image_group = 'data'
dataset = data(path)
run = Run(path)

def raw_to_OD(name):
    with h5py.File(path) as h5_file:
        abs = ma.masked_invalid(array(h5_file['data']['imagesXY_2_Flea3'][name])[0])
        probe = ma.masked_invalid(array(h5_file['data']['imagesXY_2_Flea3'][name])[1])
        bckg = ma.masked_invalid(array(h5_file['data']['imagesXY_2_Flea3'][name])[2])
        return (-log((abs - bckg)/(probe - bckg)))

A = raw_to_OD('Raw')
B = np.array(A)
# xx= np.linspace(0, 487, 488)
# yy = np.linspace(0, 647, 648)
fig_v2 = cursors_ROI(A)
clk = fig_v2.counter_tracker()
if clk < 4:
    clk = fig_v2.counter_tracker()
    print clk
elif clk == 4:
    BCK, ROI = fig_v2.get_coords()
    print BCK, ROI
plt.show()
