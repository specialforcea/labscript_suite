
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

plt.rcParams['font.size']=8

class viewer_2d(object):
    def __init__(self, z, x=None, y=None):
        """
        Shows a given array in a 2d-viewer.
        Input: z, an 2d array.
        x,y coordinters are optional.
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
        self.ROIcoords = []
        self.BCKcoords = []
        self.fig=plt.figure()
        
        #Do some layout with subplots:
        self.fig.subplots_adjust(0.05,0.05,0.98,0.98,0.1)
        self.overview=plt.subplot2grid((8,4),(0,0),rowspan=7,colspan=2)
        self.overview.pcolormesh(self.z, cmap='hot')
        self.overview.autoscale(1,'both',1)
        self.x_subplot=plt.subplot2grid((8,4),(0,2),rowspan=4,colspan=2)
        self.y_subplot=plt.subplot2grid((8,4),(4,2),rowspan=4,colspan=2)
		
        #Add widgets, to not be gc'ed, they are put in a list:
        cursor=Cursor(self.overview, useblit=True, color='k', linewidth=2 )
        but_ax=plt.subplot2grid((8,4),(7,0),colspan=1)
        reset_button=Button(but_ax,'Reset')
        but_ax2=plt.subplot2grid((8,4),(7,1),colspan=1)
        legend_button=Button(but_ax2,'Legend')
        self._widgets=[cursor,reset_button,legend_button]
        
        #Connect events
        reset_button.on_clicked(self.clear_box)
        legend_button.on_clicked(self.show_legend)
        self.fig.canvas.mpl_connect('button_press_event',self.click)
        
    
    def show_legend(self, event):        
        """Shows legend for the plots"""
        for pl in [self.x_subplot,self.y_subplot]:
            if len(pl.lines)>0:
                pl.legend()
        plt.draw()

    def clear_box(self,event):
        """Clears the ROI - BCK boxes"""
        self.ROIcoords = []
        self.BCKcoords = []
        self.overview.lines = []
        plt.draw()

    def click(self,event):
        """
        What to do, if a click on the figure happens:
            1. Get coord's
            2. Set cursor
        """
        if event.inaxes==self.overview:
			#Get nearest data position
            xpos=np.argmin(np.abs(event.xdata-self.x))
            ypos=np.argmin(np.abs(event.ydata-self.y))
		
            if event.button==1:
				#Plot ROI cursor                
                self.overview.axvline(self.x[xpos],color='black',lw=1)
                self.overview.axhline(self.y[ypos],color='black',lw=1)
                self.ROIcoords.append([xpos, ypos])
                print self.ROIcoords

            elif event.button==3:
                #Plot ROI cursor                
                self.overview.axvline(self.x[xpos],color='b',lw=1)
                self.overview.axhline(self.y[ypos],color='b',lw=1)
                self.BCKcoords.append([xpos, ypos])
                print self.BCKcoords

        #Show it
        plt.draw()

        
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
        abs = ma.masked_invalid(array(h5_file['data']['imagesYZ_1_Flea3'][name])[0])
        probe = ma.masked_invalid(array(h5_file['data']['imagesYZ_1_Flea3'][name])[1])
        bckg = ma.masked_invalid(array(h5_file['data']['imagesYZ_1_Flea3'][name])[2])
        return (-log((abs - bckg)/(probe - bckg)))
		
A = raw_to_OD('Raw')
B = np.array(A)

# xx= np.linspace(0, 487, 488)
# yy = np.linspace(0, 647, 648)
fig_v2=viewer_2d(A)

#Show it
plt.show()