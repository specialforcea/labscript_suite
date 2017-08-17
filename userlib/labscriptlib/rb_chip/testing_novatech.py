from __future__ import division, print_function

from labscript import *
from labscriptlib.common.functions import *

shared_connectiontable = labscript_import('labscriptlib.rb_chip.shared.shared_connectiontable')        

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
    
    
@stage
def prep(t):
## prepping all hardware settings
   
    
    # novatechdds9m_0 channel outputs:## -- Andika
    MOT_lock.setfreq(t, 40, units='MHz')
    MOT_lock.setamp(t, 0.5) # 0 dBm
    AOM_Raman_1.setfreq(t, 83.500, units='MHz')
    AOM_Raman_1.setamp(t, 1.0)
    AOM_Raman_2.setfreq(76500000-4*3678)
    AOM_Raman_2.setamp(1.0)
	
    RF_evap.frequency.constant(t, 30.0, units='MHz')
    RF_evap.amplitude.constant(t, 0.0)
    
    # novatechdds9m_1 channel outputs:
    uwave1.frequency.constant(t, 100.0, units='MHz')
    uwave1.amplitude.constant(t, 0.0) # -80dBm
    MOT_repump.constant(t, 0.4)
    
    return 0.03
    

@stage    
def cMOT(t):
## compressed the MOT:
	# turn off UV
	# Halfgaussramp MOT repump

    # novatechdds9m_0 channel outputs:
    #MOT_lock.frequency.customramp(t,  MOT_time, HalfGaussRamp, UVMOTFreq + ResFreq, CMOTFreq + ResFreq, CMOTCaptureWidth, samplerate=1/step_cMOT, units='MHz')
    MOT_lock.setfreq(t, 60, units = 'MHz')
    MOT_lock.setamp(t, 0.1)
    
    uwave1.frequency.constant(t, 80.0, units='MHz')
    uwave1.amplitude.constant(t, 0.5)
    RF_evap.frequency.constant(t, 10.0, units='MHz')
    RF_evap.amplitude.constant(t, 0.5)
    MOT_repump.constant(t, 0.8)
    
    return 0.03
    

@stage
def molasses(t):
    # novatechdds9m_0 channel outputs:
    #MOT_lock.frequency.customramp(t, TimeMol, ExpRamp, CMOTFreq + ResFreq , EndFreqMol + ResFreq , TauMol, samplerate=1/step_molasses, units='MHz')
    MOT_lock.setfreq(t, 20, units = 'MHz')
    MOT_lock.setamp(t, 0.3)
    MOT_repump.constant(t, 0.1)



    return 0.03
    
start()
t = 0
t += prep(t)
t += cMOT(t)
t += molasses(t)
t += 0.03

    
stop(t, min_time = t + 0.0)