from __future__ import division, print_function

from labscript import *
import sys
try:
    del sys.modules['labscriptlib.rb_chip.shared_connectiontable']
except KeyError:
    pass
import labscriptlib.rb_chip.shared_connectiontable 
    
from labscriptlib.common.functions import *


def stage(func):
    """A decorator for functions that represent stages of the experiment.
    
    Assumes first argument to function is the stage's start time, and return value is
    its duration (which is a convention we normally follow throughout labscript).
    
    The resulting decorated function simply prints the name of the stage and its start and end
    times when the function is called."""
    import functools
    @functools.wraps(func)
    def f(t, *args, **kwargs):
        duration = func(t, *args, **kwargs)
        try:
            float(duration)
        except Exception:
            raise TypeError("Stage function %s must return its duration, even if zero"%func.__name__)
        print('%s: %s sec to %s sec'%(func.__name__, str(t), str(t + duration)))
        return duration
    return f
    
    

    
    
start()
t = 0
t += prep(t)
t += UVMOT(t)
t += cMOT(t)
t += molasses(t)
t += optical_pump_1(t)
t += optical_pump_2(t)
t += magnetic_trap(t)
t += move_1(t)
t += move_2(t)
t += move_3(t)
t += move_4(t)
t += move_5(t)
t += move_6(t)
t += move_7(t)
t += move_8(t)
t += move_9(t)
t += move_10(t)
t += move_11(t)
t += move_12(t)
t += evaporate(t)
t += decompress_1(t)
t += decompress_2(t)
# t += dipole_evaporation(t)
t += TOF(t)
t += TOF_open_shutter(t)
t += repump_1(t)
t += image_1(t)
t += download_image_1(t)
t += repump_2(t)
t += image_2(t)
t += download_image_2(t)
t += repump_3(t)
t += image_3(t)
t += download_image_3(t)
t += off(t)
t += cooldown_time
stop(t)
