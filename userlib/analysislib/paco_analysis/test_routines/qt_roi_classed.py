from __future__ import division
from lyse import *
from pylab import *
from time import time
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtGui
import os
import pandas as pd
import numexpr as ne
import numpy as np
import pyqtgraph as pg
import re
import sys

class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):

        super(MainWindow, self).__init__(parent)
        self.form_widget = ROIWidget(self) 
        self.setCentralWidget(self.form_widget) 


class ROIWidget(QtGui.QWidget):

    def __init__(self, parent):        
        super(ROIWidget, self).__init__(parent)
        self.layout = QtGui.QVBoxLayout(self)

        self.button1 = QtGui.QPushButton("Button 1")
        self.layout.addWidget(self.button1)

        self.button2 = QtGui.QPushButton("Button 2")
        self.layout.addWidget(self.button2)

        self.setLayout(self.layout)


app = QtGui.QApplication([])
foo = MainWindow()
foo.show()
# sys.exit(app.exec_())