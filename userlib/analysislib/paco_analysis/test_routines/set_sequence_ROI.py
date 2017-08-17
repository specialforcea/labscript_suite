
from __future__ import division
from lyse import *
from pylab import *
from time import time
from scipy.ndimage import *
from scipy.optimize import *
from matplotlib.widgets import Cursor, Button
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
import numexpr as ne

plt.rcParams['font.size'] = 11
pd.options.display.max_colwidth = 100

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
        self.shot = plt.subplot2grid((10,3), (0,0), rowspan = 8, colspan = 3)
        self.shot.pcolormesh(self.z, cmap = 'hot')
        self.shot.autoscale(1, 'both', 1)
		
        #Cursor widget
        cursor = Cursor(self.shot, useblit = True, color = 'k', linewidth = 2)
        #Button widget
        but_ax  = plt.subplot2grid((10,3), (9,0), colspan = 1)
        reset_button   = Button(but_ax,  'Reset')
        but_ax2 = plt.subplot2grid((10,3), (9,1), colspan = 1)
        set_ROI_button = Button(but_ax2, 'Set ROI')
        but_ax3 = plt.subplot2grid((10,3), (9,2), colspan = 1)
        legend_button  = Button(but_ax3, 'Legend')
        # Widget List
        self._widgets = [cursor, reset_button, set_ROI_button, legend_button]
        
        #Connect events
        reset_button.on_clicked(self.clear_box)
        #set_ROI_button.on_clicked(self.get_coords)
        legend_button.on_clicked(self.get_coords)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
    
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
    
     
print '\nRunning %s' % os.path.basename(__file__)
t = time()

df = data()
fpstr = df['filepath'].to_string(index = False, header = False)
seq_str = re.split(r'[;,\s]\s*', fpstr)
seq_str = [fpath.encode('utf-8') for fpath in seq_str]
seq_str.remove(seq_str[0])
seq_str = np.array(seq_str)
print seq_str[0]

def raw_to_OD(name):
    with h5py.File(seq_str[0]) as h5_file:
        abs = ma.masked_invalid(array(h5_file['data']['imagesYZ_1_Flea3'][name])[0])
        probe = ma.masked_invalid(array(h5_file['data']['imagesYZ_1_Flea3'][name])[1])
        bckg = ma.masked_invalid(array(h5_file['data']['imagesYZ_1_Flea3'][name])[2])
        return (-log((abs - bckg)/(probe - bckg)))
        
A = raw_to_OD('Raw')
B = np.array(A)
fig_v2 = cursors_ROI(A)
plt.show()