from __future__ import division
from lyse import *
from pylab import *
from time import time
from pyqtgraph.Qt import QtCore, QtGui
import os
import pandas as pd
import numexpr as ne
import numpy as np
import pyqtgraph as pg
import re

pd.options.display.max_colwidth = 100
        
df = data()
fpstr = df['filepath'].to_string(index = False, header = False)
seq_str = re.split(r'[;,\s]\s*', fpstr)
seq_str = [fpath.encode('utf-8') for fpath in seq_str]
seq_str.remove(seq_str[0])
seq_str = np.array(seq_str)
print seq_str[0]

def raw_to_OD(name):
    with h5py.File(seq_str[0]) as h5_file:
        abs = ma.masked_invalid(array(h5_file['data']['imagesXY_1_Flea3'][name])[0])
        probe = ma.masked_invalid(array(h5_file['data']['imagesXY_1_Flea3'][name])[1])
        bckg = ma.masked_invalid(array(h5_file['data']['imagesXY_1_Flea3'][name])[2])
        return (-log((abs - bckg)/(probe - bckg)))

A = raw_to_OD('Raw')
arr = np.array(A)      

## Initialize GUI
app = QtGui.QApplication([])
window = QtGui.QMainWindow()
window = pg.GraphicsWindow(size=(500, 1000), border=True)
window.setWindowTitle('ROI Example')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

steps = np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
colors = ['k', 'r', 'y', 'g', 'c', 'b']
clrmp = pg.ColorMap(steps, np.array([pg.colorTuple(pg.Color(c)) for c in colors]))
lut = clrmp.getLookupTable()

text = """Data Selection From Image.\n
Drag ROI or its handles to update the selected image.
"""
ROIw1 = window.addLayout(row = 0, col = 0)
label1 = ROIw1.addLabel(text, row=0, col=0)
view1a = ROIw1.addViewBox(row = 1, col=0, lockAspect=True)
view1b = ROIw1.addViewBox(row=2, col=0, lockAspect=True)
save_ROI_button = QtGui.QPushButton('Save_ROI')
img1a = pg.ImageItem(arr, lut = lut)
img1a.setLevels([0, 1])
view1a.addItem(img1a)
img1b = pg.ImageItem(lut = lut)
img1b.setLevels([0, 1])
view1b.addItem(img1b)
view1a.disableAutoRange('xy')
view1b.disableAutoRange('xy')
view1a.autoRange()
view1b.autoRange()

recta = pg.ROI([180, 200], [200, 250])
## handles scaling horizontally around center
recta.addScaleHandle([1, 0.5], [0.5, 0.5])
recta.addScaleHandle([0, 0.5], [0.5, 0.5])
## handles scaling vertically from opposite edge
recta.addScaleHandle([0.5, 0], [0.5, 1])
recta.addScaleHandle([0.5, 1], [0.5, 0])
## handles scaling both vertically and horizontally
recta.addScaleHandle([1, 1], [0, 0])
recta.addScaleHandle([0, 0], [1, 1])
rois = []
rois.append(recta)
rois.append(pg.CircleROI([100, 100], [100, 100], pen=(1,3)))

def update(roi):
    img1b.setImage(roi.getArrayRegion(arr, img1a), levels=(0, arr.max()))
    view1b.autoRange()
    pos2 = recta.parentBounds()
    
for roi in rois:
    roi.sigRegionChanged.connect(update)
    view1a.addItem(roi)
    
update(rois[-1])

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()     