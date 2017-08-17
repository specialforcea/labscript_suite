from __future__ import division, print_function

from labscript import *
    
from labscript_devices.PulseBlaster_SP2_24_100_32k import PulseBlaster_SP2_24_100_32k
from labscript_devices.NovaTechDDS9M import NovaTechDDS9M

PulseBlaster_SP2_24_100_32k('pulseblaster_0', programming_scheme='pb_stop_programming/STOP')
ClockLine(name='pulseblaster_0_novatech_0_clock', pseudoclock=pulseblaster_0.pseudoclock, connection='flag 22')
NovaTechDDS9M(name='novatechdds9m_0', parent_device=pulseblaster_0_novatech_0_clock, com_port='com9', baud_rate = 19200, update_mode='asynchronous')
DDS(name='MOT_lock',        parent_device=novatechdds9m_0, connection='channel 0')

start()

t = 0

MOT_lock.setamp(t, 0.7077)
MOT_lock.setfreq(t, 128, 'MHz')

t += 3

t += MOT_lock.frequency.ramp(t,  10, 10, 100, samplerate=20, units='MHz')

MOT_lock.setfreq(t, 5, 'MHz')

t += 3

stop(t + 1e-3)