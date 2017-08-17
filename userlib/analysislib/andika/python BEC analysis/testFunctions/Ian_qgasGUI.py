
"""
Created on Tue Sep 17 12:49:46 2013

@author: ispielman

Modified on Wed Dec 10 11:33 2014
@author: aputra
"""

# The goal of this GUI is to give a live display of properties of my image data type
# 
# a later version will also call functions that process / fit data
# 

import matplotlib
from matplotlib.widgets import Cursor
from matplotlib.figure import Figure
import numpy
import QgasUtils

class qgasGuiObj(object):

    def __init__(self):

        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(221)
        self.ax2 = self.figure.add_subplot(222, sharey=self.ax1)
        self.ax3 = self.figure.add_subplot(223, sharex=self.ax1)
        self.ax4 = self.figure.add_subplot(224)
  
              
    def Plot(self, Image):
        x0 = numpy.array([0,0]); # Note that this is python ordered so [0] is y and [1] is x
        Slices = QgasUtils.ImageSlice(Image["xVals"], Image["yVals"], Image["OptDepth"], 20, x0,Scaled=True);

        self.ax1.clear();
        self.ax1.imshow(Image["OptDepth"], extent=[Image["ExpInf"]["x0"][1],Image["ExpInf"]["x1"][1],Image["ExpInf"]["x0"][0],Image["ExpInf"]["x1"][0]], vmax=2, vmin=-0.05, origin='lower');
        self.cur1 = Cursor(self.ax1, useblit=False, color='red', linewidth=2);            

        self.ax2.clear();
        self.ax2.plot(Slices[1][1],Slices[1][0])
        self.cur2 = Cursor(self.ax2, useblit=False, color='green', linewidth=2);            

        self.ax3.clear();
        self.ax3.plot(Slices[0][0],Slices[0][1]);
        self.cur3 = Cursor(self.ax3, useblit=False, color='black', linewidth=2);            

        self.ax4.clear();
        self.ax4.pcolormesh(Image["xVals"], Image["yVals"], Image["OptDepth"], vmax=2, vmin=-0.05);
        self.cur4 = Cursor(self.ax4, useblit=False, color='blue', linewidth=2);            

# one nice thing is that 'ui' will still contain values created during execution of the gui (including data)