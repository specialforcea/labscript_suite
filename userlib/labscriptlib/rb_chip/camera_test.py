from __future__ import division, print_function

from labscript import *
import sys
try:
    del sys.modules['labscriptlib.rb_chip.shared_connectiontable']
except KeyError:
    pass
import labscriptlib.rb_chip.shared_connectiontable 
    
start()


