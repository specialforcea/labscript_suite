from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys

class MyWindow(QWidget):
    
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.label = QLabel("Hi, I'm a window", self)
        self.button = QPushButton("&Quit", self)
        self.connect(self.button, SIGNAL("clicked()"), QCoreApplication.instance(), SLOT("quit()"))
        lay = QVBoxLayout()
        
        lay.addWidget(self.label)
        lay.addWidget(self.button)
        self.setLayout(lay)

app = QApplication(sys.argv)
mw = MyWindow()
mw.show()
sys.exit(app.exec_())