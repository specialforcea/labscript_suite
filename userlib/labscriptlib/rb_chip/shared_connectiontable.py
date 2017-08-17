from __future__ import division

from labscript import *
from labscript_devices.PulseBlaster_SP2_24_100_32k import PulseBlaster_SP2_24_100_32k
from labscript_devices.NI_PCI_6733 import NI_PCI_6733
from labscript_devices.NI_USB_6343 import NI_USB_6343
from labscript_devices.NI_DAQmx import NI_DAQmx
from labscript_devices.NovaTechDDS9M import NovaTechDDS9M
from labscript_utils.unitconversions import BidirectionalCoilDriver
from labscript_devices.Camera import Camera

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


# NI_DAQmx(name='ni_pci_6733_0', parent_device=pulseblaster_0_ni_pci_clock,
#         MAX_name='Dev1',
#         clock_terminal='/Dev1/RTSI0', 
#         num_AO=8,
#         sample_rate_AO=700e3,
#         num_DO=8,
#         sample_rate_DO=2e6,
#         num_AI=0,
#         clock_terminal_AI='/Dev1/PFI0',
#         mode_AI='labscript', # 'labscript', 'gated', 'triggered'
#         sample_rate_AI=1000,
#         num_PFI=0
#)

# NI_PCI_6733(name='ni_pci_6733_0', parent_device=pulseblaster_0_ni_pci_clock, clock_terminal='/Dev1/RTSI0', MAX_name='Dev1')

# AnalogOut(name='MOT_repump',       parent_device=ni_pci_6733_0, connection='ao0')
# AnalogOut(name='probe_repump',     parent_device=ni_pci_6733_0, connection='ao1')
# AnalogOut(name='MOT_cooling',      parent_device=ni_pci_6733_0, connection='ao2')
# AnalogOut(name='optical_pumping',  parent_device=ni_pci_6733_0, connection='ao3')
# AnalogOut(name='probe_1',          parent_device=ni_pci_6733_0, connection='ao4')
# AnalogOut(name='probe_2',          parent_device=ni_pci_6733_0, connection='ao5')
# AnalogOut(name='dipole_intensity', parent_device=ni_pci_6733_0, connection='ao6')
# AnalogOut(name='dipole_split',     parent_device=ni_pci_6733_0, connection='ao7')

# DigitalOut(name='DO0_0',        parent_device=ni_pci_6733_0, connection='port0/line0')
# DigitalOut(name='DO0_1',        parent_device=ni_pci_6733_0, connection='port0/line1')
# DigitalOut(name='DO0_2',        parent_device=ni_pci_6733_0, connection='port0/line2')
# DigitalOut(name='DO0_3',        parent_device=ni_pci_6733_0, connection='port0/line3')
# DigitalOut(name='RF_TTL_1',     parent_device=ni_pci_6733_0, connection='port0/line4')
# DigitalOut(name='RF_TTL_2',     parent_device=ni_pci_6733_0, connection='port0/line5')
# DigitalOut(name='DO0_6',        parent_device=ni_pci_6733_0, connection='port0/line6')
# DigitalOut(name='DO0_7',        parent_device=ni_pci_6733_0, connection='port0/line7')

# NI_PCI_6733(name='ni_pci_6733_1', parent_device=pulseblaster_0_ni_pci_clock, clock_terminal='/Dev2/RTSI0', MAX_name='Dev2')

# AnalogOut(name='RF_mixer1',               parent_device=ni_pci_6733_1, connection='ao0')
# AnalogOut(name='RF_mixer2',               parent_device=ni_pci_6733_1, connection='ao1')
# AnalogOut(name='RF_mixer3',               parent_device=ni_pci_6733_1, connection='ao2')              
# AnalogOut(name='RF_mixer4',               parent_device=ni_pci_6733_1, connection='ao3')
# AnalogOut(name='microwave_attenuator',    parent_device=ni_pci_6733_1, connection='ao4')
# AnalogOut(name='uwave_power_1',           parent_device=ni_pci_6733_1, connection='ao5')
# AnalogOut(name='uwave_power_2',           parent_device=ni_pci_6733_1, connection='ao6')
# AnalogOut(name='fine_zbias',              parent_device=ni_pci_6733_1, connection='ao2',
          # unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': kepco_servo_scaling})

# DigitalOut(name='DO1_0',        parent_device=ni_pci_6733_1, connection='port0/line0')
# DigitalOut(name='DO1_1',        parent_device=ni_pci_6733_1, connection='port0/line1')
# DigitalOut(name='DO1_2',        parent_device=ni_pci_6733_1, connection='port0/line2')
# DigitalOut(name='DO1_3',        parent_device=ni_pci_6733_1, connection='port0/line3')
# DigitalOut(name='DO1_4',        parent_device=ni_pci_6733_1, connection='port0/line4')
# DigitalOut(name='DO1_5',        parent_device=ni_pci_6733_1, connection='port0/line5')
# DigitalOut(name='DO1_6',        parent_device=ni_pci_6733_1, connection='port0/line6')
# DigitalOut(name='DO1_7',        parent_device=ni_pci_6733_1, connection='port0/line7')

#NI_DAQmx(name='ni_usb_6343_0', parent_device=pulseblaster_0_ni_usb_0_clock,
#         MAX_name='Dev3',
#         clock_terminal='/Dev3/PFI0', 
#         num_AO=4,
#         sample_rate_AO=500e3,
#         num_DO=32,
#         sample_rate_DO=500e3,
#         num_AI=32,
#         clock_terminal_AI='/Dev3/PFI0',
#         mode_AI='labscript', # 'labscript', 'gated', 'triggered'
#         sample_rate_AI=1000,
#         num_PFI=0
#)

NI_USB_6343(name='ni_usb_6343_0', parent_device=pulseblaster_0_ni_usb_0_clock, clock_terminal='/Dev3/PFI0', MAX_name='Dev3')

AnalogOut(name='transport_current_1',     parent_device=ni_usb_6343_0, connection='ao0',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_1})
AnalogOut(name='transport_current_2',     parent_device=ni_usb_6343_0, connection='ao1',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_2})
AnalogOut(name='transport_current_3',     parent_device=ni_usb_6343_0, connection='ao2',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_3})
AnalogOut(name='transport_current_4',     parent_device=ni_usb_6343_0, connection='ao3',
          unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': transport_scaling_4})

# AnalogIn(name='transport_current_1_AI', parent_device=ni_usb_6343_0, connection='ai0')
# AnalogIn(name='transport_current_2_AI', parent_device=ni_usb_6343_0, connection='ai1')
# AnalogIn(name='transport_current_3_AI', parent_device=ni_usb_6343_0, connection='ai2')
# AnalogIn(name='transport_current_4_AI', parent_device=ni_usb_6343_0, connection='ai3')
          
DigitalOut(name='curent_supply_1_select_line0', parent_device=ni_usb_6343_0, connection='port0/line0')
DigitalOut(name='curent_supply_1_select_line1', parent_device=ni_usb_6343_0, connection='port0/line1')
DigitalOut(name='curent_supply_1_enable',       parent_device=ni_usb_6343_0, connection='port0/line2')

DigitalOut(name='curent_supply_2_select_line0', parent_device=ni_usb_6343_0, connection='port0/line3')
DigitalOut(name='curent_supply_2_select_line1', parent_device=ni_usb_6343_0, connection='port0/line4')
DigitalOut(name='curent_supply_2_enable',       parent_device=ni_usb_6343_0, connection='port0/line5')

DigitalOut(name='curent_supply_3_select_line0', parent_device=ni_usb_6343_0, connection='port0/line6')
DigitalOut(name='curent_supply_3_select_line1', parent_device=ni_usb_6343_0, connection='port0/line7')
DigitalOut(name='curent_supply_3_enable',       parent_device=ni_usb_6343_0, connection='port0/line8')

DigitalOut(name='curent_supply_4_select_line0', parent_device=ni_usb_6343_0, connection='port0/line9')
DigitalOut(name='curent_supply_4_select_line1', parent_device=ni_usb_6343_0, connection='port0/line10')
DigitalOut(name='curent_supply_4_enable',       parent_device=ni_usb_6343_0, connection='port0/line11')
DigitalOut(name='bogus_DO_DEV3_12',             parent_device=ni_usb_6343_0, connection='port0/line12')

DigitalOut(name='_CycleX_AI_pause_line_unused', parent_device=ni_usb_6343_0, connection='port0/line15')


# NI_USB_6343(name='ni_usb_6343_1', parent_device=pulseblaster_0_ni_usb_1_clock, clock_terminal='/Dev4/PFI0', MAX_name='Dev4')

# AnalogOut(name='x_bias',  parent_device=ni_usb_6343_1, connection='ao0',
          # unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': kepco_servo_scaling})
# AnalogOut(name='y_bias',  parent_device=ni_usb_6343_1, connection='ao1',
          # unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': kepco_y_scaling})
# AnalogOut(name='z_bias',  parent_device=ni_usb_6343_1, connection='ao2',
          # unit_conversion_class=BidirectionalCoilDriver, unit_conversion_parameters={'slope': kepco_z_scaling})
# AnalogOut(name='offset',  parent_device=ni_usb_6343_1, connection='ao3')

# DigitalOut(name='kepco_enable_0', parent_device=ni_usb_6343_1, connection='port0/line0')
# DigitalOut(name='kepco_enable_1', parent_device=ni_usb_6343_1, connection='port0/line1')

#NI_USB_6343(name='ni_usb_6343_2', parent_device=pulseblaster_0_ni_usb_2_clock, clock_terminal='/DevTiSapp/PFI0', MAX_name='DevTiSapp')

# NI_DAQmx(name='ni_usb_6343_2', parent_device=pulseblaster_0_ni_usb_2_clock,
         # MAX_name='DevTiSapp',
         # clock_terminal='/DevTiSapp/PFI0', 
         # num_AO=4,
         # sample_rate_AO=700e3,
         # num_DO=32,
         # sample_rate_DO=2e6,
         # num_AI=8,
         # clock_terminal_AI='/DevTiSapp/PFI3',
         # mode_AI='labscript', # 'labscript', 'gated', 'triggered'
         # sample_rate_AI=1000, 
         # num_PFI=0
# )

# AnalogOut(name='mixer_raman_1',     parent_device=ni_usb_6343_2, connection='ao0')
# AnalogOut(name='mixer_raman_2',     parent_device=ni_usb_6343_2, connection='ao1')
# AnalogOut(name='AOM_y_beam',      parent_device=ni_usb_6343_2, connection='ao2')
# AnalogOut(name='AOM_green_beam',  parent_device=ni_usb_6343_2, connection='ao3')

# DigitalOut(name='raman_1', parent_device=ni_usb_6343_2, connection='port0/line0')
# DigitalOut(name='raman_2', parent_device=ni_usb_6343_2, connection='port0/line1')
# DigitalOut(name='green_beam', parent_device=ni_usb_6343_2, connection='port0/line2')
# DigitalOut(name='shutter_raman1', parent_device=ni_usb_6343_2, connection='port0/line3')
# DigitalOut(name='shutter_raman2', parent_device=ni_usb_6343_2, connection='port0/line4')
# DigitalOut(name='shutter_greenbeam', parent_device=ni_usb_6343_2, connection='port0/line5')

# NovaTechDDS9M(name='novatechdds9m_0', parent_device=pulseblaster_0_novatech_0_clock, com_port='com9',
              # baud_rate = 19200, update_mode='asynchronous')

# DDS(name='MOT_lock',            parent_device=novatechdds9m_0, connection='channel 0')
# DDS(name='AOM_Raman_1',         parent_device=novatechdds9m_0, connection='channel 1')
# StaticDDS(name='AOM_Raman_2',   parent_device=novatechdds9m_0, connection='channel 2')
# StaticDDS(name='Nov_0_3',       parent_device=novatechdds9m_0, connection='channel 3')

# NovaTechDDS9M(name='novatechdds9m_1', parent_device=pulseblaster_0_novatech_1_clock, com_port='com7',
              # baud_rate = 19200, update_mode='asynchronous')

# DDS(name='RF_evap',         parent_device=novatechdds9m_1, connection='channel 0')
# DDS(name='uwave1',          parent_device=novatechdds9m_1, connection='channel 1')
# StaticDDS(name='RF2',       parent_device=novatechdds9m_1, connection='channel 2')
# StaticDDS(name='uwave2',    parent_device=novatechdds9m_1, connection='channel 3')

#
# Cameras
#

#Camera(name='XY_1_Flea3', parent_device=camera_trigger_1, connection='trigger', BIAS_port=1024, 
#         serial_number=0x00B09D0100AC001B, SDK='IMAQdx_Flea3_Firewire', effective_pixel_size=5.9e-6, exposuretime=10e-3, 
#         orientation = 'xy', trigger_edge_type='falling')
        
# Camera(name='XY_2_Flea3', parent_device=camera_trigger_1, connection='trigger', BIAS_port=1025, 
        # serial_number=0x00B09D0100AC001C, SDK='IMAQdx_Flea3_Firewire', effective_pixel_size=5.9e-6, exposuretime=10e-3, 
        # orientation = 'xy')
        
# Camera(name='XY_3_Flea3', parent_device=camera_trigger_1, connection='trigger', BIAS_port=1026, 
        # serial_number=0x00B09D0100AC001D, SDK='IMAQdx_Flea3_Firewire', effective_pixel_size=5.9e-6, exposuretime=10e-3, 
        # orientation = 'xy')		#effective_pixel_size = 5.6e-6 over magnification?
        
# Setup for 12 bit images acquired as U16
IMAQdx_properties = [
["CameraAttributes::Sharpness::Mode", "Off"],
["CameraAttributes::Gamma::Mode", "Off"],
["CameraAttributes::FrameRate::Mode", "Auto"],
["CameraAttributes::Brightness::Mode", "Absolute"],
["CameraAttributes::Brightness::Value", "0.0"],
["CameraAttributes::AutoExposure::Mode", "Off"],
["CameraAttributes::TriggerDelay::Mode", "Off"],
["CameraAttributes::Trigger::TriggerActivation", "Level Low"], # 0 for enum
["CameraAttributes::Trigger::TriggerMode", "Mode 1"], # 1 for enum, Bulb Mode
["CameraAttributes::Trigger::TriggerSource", "Source 0"],
["CameraAttributes::Trigger::TriggerParameter", "0"],
["CameraAttributes::Shutter::Mode", "Absolute"],
["CameraAttributes::Shutter::Value", "10e-3"],
["CameraAttributes::Gain::Mode", "Absolute"], # Must be before Gain::Value
["CameraAttributes::Gain::Value", "10"],
["AcquisitionAttributes::Timeout", "120000"],
["AcquisitionAttributes::Speed", "400 Mbps"], # 2 for enum, options 100,200,400,800
["AcquisitionAttributes::VideoMode", "Format 7, Mode 0, 648 x 488"], ### # 13 change to Format 7, Mode 1, 324 x 244 or Mode 5 #14 instead of 13
["AcquisitionAttributes::Width", "648"],	#324 ## change here
["AcquisitionAttributes::Height", "488"],	#244 ## change here
["AcquisitionAttributes::ShiftPixelBits", "true"],
["AcquisitionAttributes::SwapPixelBytes", "false"],
["AcquisitionAttributes::BitsPerPixel", "12-bit"],
["AcquisitionAttributes::PixelFormat", "Mono 16"] # Needs to be near the end for some reason
]

# Could not find "PixelSignedness" which wants to be "Unsigned", 0 for enum

# Video Modes:
# 0-6: Mono 8, 640x480, 1.88 to 120 fps
# 7-12: Mono 16, 640x480, 1.88 to 60 fps
# *9: 7.5 fps (max at 100 Mbps) (don't set PixelFormat for this)
# *13: Format 7, Mode 0, 648 x 488
# 14: Format 7, Mode 1, 324 x 244
# 15: Format 7, Mode 7, 648 x 488

# These modes can be "discovered" with Enumate Video Modes.vi.
# The stared "*" ones above work best for our purposes.

# Camera(name='YZ_1_Flea3', parent_device=camera_trigger_1, connection='trigger', trigger_edge_type='falling',
         # BIAS_port=1027, SDK='IMAQdx_Flea3_Firewire', effective_pixel_size=5.9e-6, orientation = 'yz', 
         # exposure_time=100e-6,
         # serial_number=0x00B09D0100AC001B, 
         # added_properties=IMAQdx_properties
          # )
        
    
class ServoOutputSelect(object):
    """Object to abstract the output selection on one of the current servos"""
    def __init__(self, line0, line1):
        # Line 0 is most significant bit, line 1 is least significant bit.
        self.line0 = line0
        self.line1 = line1
        
    def __call__(self, t, number):
        """Given an output number from 0 - 3,
        sets the two output lines with the corresponding binary number"""
        bits = bin(number)[2:]
        if int(bits[0]):
            line0.go_high(t)
        else:
            line0.go_low(t)
        if int(bits[1]):
            line1.go_high(t)
        else:
            line1.go_low(t)
            
# # convenience functions for selecting servo outputs:
# servo_1_select_output = ServoOutputSelect(curent_supply_1_select_line0, curent_supply_1_select_line1)
# servo_2_select_output = ServoOutputSelect(curent_supply_2_select_line0, curent_supply_2_select_line1)
# servo_3_select_output = ServoOutputSelect(curent_supply_3_select_line0, curent_supply_3_select_line1)
# servo_4_select_output = ServoOutputSelect(curent_supply_4_select_line0, curent_supply_4_select_line1)
