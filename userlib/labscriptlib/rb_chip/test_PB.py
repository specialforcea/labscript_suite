from __future__ import division, print_function

from labscript import *
from labscript_devices.PulseBlaster_SP2_24_100_32k import PulseBlaster_SP2_24_100_32k
from labscript_devices.NI_USB_6343 import NI_USB_6343

PulseBlaster_SP2_24_100_32k('pulseblaster_0', programming_scheme='pb_stop_programming/STOP')

DigitalOut(name='AOM_MOT_repump',          parent_device=pulseblaster_0.direct_outputs, connection='flag 0')
DigitalOut(name='shutter_MOT_repump',      parent_device=pulseblaster_0.direct_outputs, connection='flag 1')
Trigger(name='camera_trigger_1',           parent_device=pulseblaster_0.direct_outputs, connection='flag 3',
        trigger_edge_type='falling')
DigitalOut(name='AOM_MOT_cooling',         parent_device=pulseblaster_0.direct_outputs, connection='flag 4')
DigitalOut(name='shutter_MOT_cooling',     parent_device=pulseblaster_0.direct_outputs, connection='flag 5')
DigitalOut(name='AOM_optical_pumping',     parent_device=pulseblaster_0.direct_outputs, connection='flag 6')
DigitalOut(name='shutter_optical_pumping', parent_device=pulseblaster_0.direct_outputs, connection='flag 7')
DigitalOut(name='AOM_probe_1',             parent_device=pulseblaster_0.direct_outputs, connection='flag 8')
DigitalOut(name='shutter_z',               parent_device=pulseblaster_0.direct_outputs, connection='flag 9')
DigitalOut(name='AOM_probe_2',             parent_device=pulseblaster_0.direct_outputs, connection='flag 10')
DigitalOut(name='shutter_x',               parent_device=pulseblaster_0.direct_outputs, connection='flag 11')
DigitalOut(name='UV_LED',                  parent_device=pulseblaster_0.direct_outputs, connection='flag 12')
Trigger(name='camera_trigger_0',           parent_device=pulseblaster_0.direct_outputs, connection='flag 13',
        trigger_edge_type='falling')
DigitalOut(name='vacuum_shutter',          parent_device=pulseblaster_0.direct_outputs, connection='flag 14')
DigitalOut(name='RF_TTL',                  parent_device=pulseblaster_0.direct_outputs, connection='flag 15')
DigitalOut(name='shutter_top_repump',      parent_device=pulseblaster_0.direct_outputs, connection='flag 16')
DigitalOut(name='dipole_switch',           parent_device=pulseblaster_0.direct_outputs, connection='flag 17')

ClockLine(name='pulseblaster_0_ni_pci_clock',     pseudoclock=pulseblaster_0.pseudoclock, connection='flag 23')
ClockLine(name='pulseblaster_0_ni_usb_0_clock',   pseudoclock=pulseblaster_0.pseudoclock, connection='flag 21')
ClockLine(name='pulseblaster_0_ni_usb_1_clock',   pseudoclock=pulseblaster_0.pseudoclock, connection='flag 20')
ClockLine(name='pulseblaster_0_ni_usb_2_clock',   pseudoclock=pulseblaster_0.pseudoclock, connection='flag 18')
ClockLine(name='pulseblaster_0_novatech_0_clock', pseudoclock=pulseblaster_0.pseudoclock, connection='flag 22')
ClockLine(name='pulseblaster_0_novatech_1_clock', pseudoclock=pulseblaster_0.pseudoclock, connection='flag 2')   

NI_USB_6343(name='ni_usb_6343_0', parent_device=pulseblaster_0_ni_usb_0_clock, clock_terminal='/Dev3/PFI0', MAX_name='Dev3')

AnalogOut(name='transport_current_1',     parent_device=ni_usb_6343_0, connection='ao0',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_1})
AnalogOut(name='transport_current_2',     parent_device=ni_usb_6343_0, connection='ao1',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_2})
AnalogOut(name='transport_current_3',     parent_device=ni_usb_6343_0, connection='ao2',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_3})
AnalogOut(name='transport_current_4',     parent_device=ni_usb_6343_0, connection='ao3',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_4})
   
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
    
@stage    
def UV(t):
    # pulseblaster_0 outputs:
    UV_LED.go_high(t)
    shutter_MOT_cooling.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, UVMOTCurrent, units='A')
    
    return 1
    
@stage
def off(t):
    # pulseblaster_0 outputs:
    UV_LED.go_low(t)
    shutter_MOT_cooling.go_low(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, 0, units='A')
        
    return 1
    
start()
t = 0
t += UV(t)
t += off(t)
stop(t)
