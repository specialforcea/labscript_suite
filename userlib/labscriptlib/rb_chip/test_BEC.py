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
    
    
@stage
def prep(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_high(t)
    shutter_MOT_repump.go_high(t)
    camera_trigger_0.go_high(t)    
    AOM_MOT_cooling.go_high(t)
    shutter_MOT_cooling.go_high(t)
    AOM_probe_1.go_high(t)
    #camera_trigger_1.go_high(t)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_enable.go_high(t)
    
    # ni_usb_6343_1 digital outputs:
    kepco_enable_0.go_high(t)
    kepco_enable_1.go_high(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t,  UVMOTRepump)
    MOT_cooling.constant(t, UVMOTAmplitude)
    probe_1.constant(t,     ProbeInt)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, UVMOTCurrent, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 2.11*UVxShim, units='A')
    y_bias.constant(t,      UVyShim, units='A')
    z_bias.constant(t, -0.8*UVzShim, units='A')
    
    # ni_usb_6343_1 analog outputs:
    mixer_raman_1.constant(t, 0.31)
    mixer_raman_1.constant(t+0.01, 0.0)
    shutter_greenbeam.go_high(t)
    shutter_greenbeam.go_low(t+0.01)
    
    
    # novatechdds9m_0 channel outputs:
    MOT_lock.setfreq(t,  (ResFreq + UVMOTFreq)*MHz)
    MOT_lock.setamp(t,  0.7077) # 0 dBm
    RF_evap.setfreq(t,  30*MHz)
    
    return 30*ms
    

@stage    
def UVMOT(t):
    # pulseblaster_0 outputs:
    UV_LED.go_high(t)
    return UV_MOT_time
    
    
@stage    
def cMOT(t):
    # pulseblaster_0 outputs:
    UV_LED.go_low(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.customramp(t,  MOT_time, HalfGaussRamp, RepumpMOT, RepumpCMOT, CMOTCaptureWidth, samplerate=1/step_cMOT)
    MOT_cooling.constant(t,   MOTAmplitude)
        
    # ni_usb_6343_0 analog outputs:
    transport_current_2.customramp(t, MOT_time, HalfGaussRamp, MOTCurrent, CMOTCurrent, CMOTCaptureWidth, samplerate=1/step_cMOT, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 2.11*MOTxShim, units='A')
    y_bias.constant(t,      MOTyShim, units='A')
    z_bias.constant(t, -0.8*MOTzShim, units='A')

    # novatechdds9m_0 channel outputs:
    MOT_lock.frequency.customramp(t,  MOT_time, HalfGaussRamp, MOTFreq + ResFreq, CMOTFreq + ResFreq, CMOTCaptureWidth, samplerate=1/step_cMOT, units='MHz')
     
    return MOT_time
    

@stage
def molasses(t):
    # pulseblaster_0 outputs:
    shutter_optical_pumping.go_high(t)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_enable.go_low(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t, RepumpMol)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, MOTCurrent, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 2.11*0.44, units='A')
    y_bias.constant(t, 0.42, units='A')
    z_bias.constant(t, -0.8*-0.04, units='A')
    
    # novatechdds9m_0 channel outputs:
    MOT_lock.frequency.customramp(t, TimeMol, ExpRamp, StartFreqMol + ResFreq , EndFreqMol + ResFreq , TauMol, samplerate=1/step_molasses, units='MHz')
    
    return TimeMol
    

@stage
def optical_pump_1(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_low(t)
    AOM_MOT_cooling.go_low(t)
    shutter_MOT_cooling.go_low(t)

    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t, RepumpOpPump)
    MOT_cooling.constant(t, 0)
    optical_pumping.constant(t, OpPumpAmplitude)
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 2.11*OpxShim, units='A')
    y_bias.constant(t, OpyShim, units='A')
    z_bias.constant(t, -0.8*OpzShim, units='A')
    
    # novatechdds9m_0 channel outputs:
    # Redundant: already set in previous stage (is final value of ramp). Leaving in anyway:
    MOT_lock.frequency.constant(t, ResFreq + EndFreqMol, units='MHz')
    
    return TimePrePump

    
@stage
def optical_pump_2(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_high(t)
    AOM_optical_pumping.go_high(t)
    
    return TimePump
    

@stage
def magnetic_trap(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_low(t)
    shutter_MOT_repump.go_low(t)
    AOM_optical_pumping.go_low(t)
    shutter_optical_pumping.go_low(t)
    AOM_probe_1.go_low(t)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_enable.go_high(t)
    curent_supply_3_enable.go_high(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t, 0)
    optical_pumping.constant(t, 0)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.customramp(t, TrapTime,  HalfGaussRamp, MOTCaptureCurrent, IM, CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.customramp(t, TrapTime, HalfGaussRamp,  2.11*CapxShim, 2.11*0.44, CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    y_bias.customramp(t, TrapTime, HalfGaussRamp,  CapyShim, 0.42,           CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    z_bias.customramp(t, TrapTime, HalfGaussRamp, -0.8*CapzShim, -0.8*-0.04, CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    
    return TrapTime
    

@stage
def move_1(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_high(t)
    AOM_probe_1.go_high(t)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_4_enable.go_high(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t, 0.8)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove01, Poly4Asymmetric, IPM, WidthPM, SkewPM, 1, Cf=[Cf5, AccelFrac, 0, Vel0], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove01, PolyHalf2, 0, IM, WidthM, SkewM,          Cf=[Cf5, AccelFrac, 0, Vel0], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove01, Poly4,     0, IMB, WidthMB, SkewMB,       Cf=[Cf5, AccelFrac, 0, Vel0], samplerate=1/step_magnetic_transport, units='A')                               
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t,  2.11*0.44, units='A')
    y_bias.constant(t,  0.42, units='A')
    z_bias.constant(t, -0.8*-0.04, units='A')                               
    
    return TimeMove01


@stage
def move_2(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_1_select_line0.go_high(t)
    curent_supply_1_enable.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.constant(t, 0, units='A')
    transport_current_2.customramp(t, TimeMove02, PolyHalf2, 1, IM, WidthM, SkewM,       Cf=[Cf2, d1, Vel0, Vel1], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove02, Poly4,     1, IMB, WidthMB, SkewMB,    Cf=[Cf2, d1, Vel0, Vel1], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove02, PolyExp,   0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel0, Vel1], samplerate=1/step_magnetic_transport, units='A')
    
    return TimeMove02
    
    
@stage
def move_3(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_select_line0.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove03, PolyExp, 0, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel1, Vel2], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.constant(t, 0, units='A')
    transport_current_3.customramp(t, TimeMove03, Poly4,   2, IMB, WidthMB, SkewMB,    Cf=[Cf2, d1, Vel1, Vel2], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove03, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel1, Vel2], samplerate=1/step_magnetic_transport, units='A')

    return TimeMove03
    

@stage
def move_4(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_3_select_line0.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove04, PolyExp, 1, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel2, Vel3], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove04, PolyExp, 0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel2, Vel3], samplerate=1/step_magnetic_transport, units='A')                      
    transport_current_3.constant(t, 0, units='A')
    transport_current_4.customramp(t, TimeMove04, PolyExp, 2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel2, Vel3], samplerate=1/step_magnetic_transport, units='A')

    return TimeMove04 
    

@stage
def move_5(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_4_select_line0.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove05, PolyExp, 2, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel3, Vel4], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove05, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel3, Vel4], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove05, PolyExp, 0, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel3, Vel4], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.constant(t, 0, units='A')
    
    return TimeMove05

    
@stage
def move_6(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_1_select_line0.go_low(t)
    curent_supply_1_select_line1.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.constant(t, 0, units='A')
    transport_current_2.customramp(t, TimeMove06, PolyExp, 2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel4, Vel5], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove06, PolyExp, 1, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel4, Vel5], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove06, PolyExp, 0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel4, Vel5], samplerate=1/step_magnetic_transport, units='A')
    
    return TimeMove06
    

@stage
def move_7(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_select_line0.go_low(t)
    curent_supply_2_select_line1.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove07, PolyExp, 0, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel5, Vel6], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.constant(t, 0, units='A')
    transport_current_3.customramp(t, TimeMove07, PolyExp, 2, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel5, Vel6], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove07, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel5, Vel6], samplerate=1/step_magnetic_transport, units='A')

    return TimeMove07
 

@stage
def move_8(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_3_select_line0.go_low(t)
    curent_supply_3_select_line1.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove08, PolyExp, 1, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel6, Vel7], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove08, PolyExp, 0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel6, Vel7], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.constant(t, 0, units='A')
    transport_current_4.customramp(t, TimeMove08, PolyExp, 2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel6, Vel7], samplerate=1/step_magnetic_transport, units='A')

    return TimeMove08
     

@stage
def move_9(t):     
    # ni_usb_6343_0 digital outputs:
    curent_supply_4_select_line0.go_low(t)
    curent_supply_4_enable.go_low(t)
    
    # ni_usb_6343_1 digital outputs:
    kepco_enable_0.go_low(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.customramp(t, TimeMove09, PolyExp, 2, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel7, Vel8], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove09, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel7, Vel8], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove09, Poly4,   0, IBT, WidthBT, SkewBT,    Cf=[Cf2, d1, Vel7, Vel8], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.constant(t, -1, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.customramp(t, TimeMove09, LineRamp,  2.11*0.44, TpxShim,     samplerate=1/step_magnetic_transport, units='A')
    y_bias.customramp(t, TimeMove09, LineRamp,  0.42, TpyShimInitial,   samplerate=1/step_magnetic_transport, units='A')
    z_bias.customramp(t, TimeMove09, LineRamp, -0.8*-0.04, a0*TpzShim,  samplerate=1/step_magnetic_transport, units='A')
    
    return TimeMove09
    
    
@stage
def move_10(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_1_select_line1.go_low(t)
    curent_supply_1_enable.go_low(t)
    curent_supply_4_select_line0.go_high(t)
    curent_supply_4_select_line1.go_high(t)
    curent_supply_4_enable.go_high(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.constant(t, -1, units='A')
    transport_current_2.customramp(t, TimeMove10, PolyExp,   2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel8, Vel9], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove10, Poly4,     1, IBT, WidthBT, SkewBT,    Cf=[Cf2, d1, Vel8, Vel9], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove10, PolyHalf1, 0, IFT, WidthFT, SkewFT,    Cf=[Cf2, d1, Vel8, Vel9], samplerate=1/step_magnetic_transport, units='A')
    
    # ni_usb_6343_1 analog outputs:
    # These lines redundant: already set in previous stage (are final values of ramps). Leaving in anyway:
    x_bias.constant(t, TpxShim, units='A')
    y_bias.constant(t, TpyShimInitial, units='A')
    z_bias.constant(t, a0*TpzShim, units='A')
    
    return TimeMove10
    
    
@stage
def move_11(t):
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_select_line1.go_low(t)
    curent_supply_2_enable.go_low(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_1.constant(t,  0, units='A')
    transport_current_2.constant(t, -1, units='A')
    transport_current_3.customramp(t, TimeMove11, Poly4,     2, IBT, WidthBT, SkewBT, Cf=[Cf2, d2, Vel9, 0], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove11, PolyHalf1, 1, IFT, WidthFT, SkewFT, Cf=[Cf2, d2, Vel9, 0], samplerate=1/step_magnetic_transport, units='A')
       
    return TimeMove11    
    
    
@stage
def move_12(t):
    
    # ni_usb_6343_0 analog outputs:
    transport_current_3.constant(t, -1, units='A')
    transport_current_4.customramp(t, TimeMove12, LineRamp, IFT, TopTrapCurrent, samplerate=1/step_magnetic_transport, units='A')
    
    # ni_usb_6343_1 analog outputs:
    y_bias.customramp(t, TimeMove12, LineRamp, TpyShimInitial, TpyShimTrap, samplerate=1/step_magnetic_transport, units='A')
    
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.constant(t, DipoleInt)
    dipole_split.constant(t, DipoleSplit)
    
    return TimeMove12
    

@stage
def evaporate(t):
    # pulseblaster_0 outputs:
    RF_TTL.go_high(t)
    dipole_switch.go_high(t)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_3_select_line1.go_low(t)
    curent_supply_3_enable.go_low(t)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, 0, units='A')
    transport_current_3.constant(t, 0, units='A')
    
    # ni_pci_6733_1 analog outputs:
    RF_mixer1.constant(t, -9.000)
    
    # novatechdds9m_0 channel outputs:
    RF_evap.frequency.customramp(t, RFevapTime, LineRamp, StartRF, EndRF, samplerate=1/step_RF_evap, units='MHz')
    RF_evap.setamp(t, 1.000)
    
    return RFevapTime
    
    
@stage
def decompress_1(t):
    # ni_usb_6343_0 analog outputs:
    transport_current_4.customramp(t, DecompTime1, LineRamp, TopTrapCurrent, MiddleCurrent, samplerate=1/step_decompress, units='A')
    
    # ni_usb_6343_1 analog outputs:
    y_bias.customramp(t, DecompTime1, LineRamp, TpyShimTrap, TpyShimTrap1, samplerate=1/step_decompress, units='A')
    
    return DecompTime1

    
@stage
def set_novatech(t):
    # novatechdds9m_0 channel outputs:
    MOT_lock.frequency.customramp(t, TimeSetNovatech, LineRamp, (ResFreq+ImageFreq+0.001), (ResFreq+ImageFreq), samplerate=1/step_set_novatech, units='MHz')
    RF_evap.setamp(t, 0.7077)
    return TimeSetNovatech
    
    
@stage
def decompress_2(t):
    # ni_usb_6343_0 analog outputs:
    transport_current_4.customramp(t, DecompTime2, LineRamp, MiddleCurrent, FinalCurrent, samplerate=1/step_decompress, units='A')
    
    # ni_usb_6343_1 analog outputs:
    y_bias.customramp(t, DecompTime2, LineRamp, TpyShimTrap1, TpyShimTrap2, samplerate=1/step_decompress, units='A')
    
    return DecompTime2

    
@stage
def dipole_evaporation_1(t):
    # pulseblaster_0 outputs:
    RF_TTL.go_low(t)
    
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.customramp(t, EvapTime, ExpRamp, DipoleInt, DipoleInt2, dipoleEvapTau, samplerate=1/step_dipole_evap)
    dipole_split.customramp(t, EvapTime, ExpRamp, DipoleSplit, FinalDipoleSplit, dipoleEvapTau, samplerate=1/step_dipole_evap)
    
    # ni_pci_6733_1 analog outputs:
    RF_mixer1.constant(t, 0)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_4.customramp(t, EvapTime, ExpRamp, FinalCurrent, 0, dipoleEvapTau, samplerate=1/step_dipole_evap, units='A')
    
    # ni_usb_6343_1 analog outputs:
    y_bias.customramp(t, EvapTime, ExpRamp, TpyShimTrap2, TpyShimTrap3, dipoleEvapTau, samplerate=1/step_dipole_evap, units='A')
    
    return EvapTime
    

@stage
def dipole_evaporation_2(t):
    
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.customramp(t, EvapTime_2, ExpRamp, DipoleInt2, DipoleInt3, dipoleEvapTau2, samplerate=1/step_dipole_evap)
    dipole_split.constant(t, FinalDipoleSplit)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_4_select_line0.go_low(t)
    curent_supply_4_select_line1.go_low(t)
    curent_supply_4_enable.go_low(t)
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, TpxShimEnd, units='A')
    y_bias.constant(t, TpyShimEnd, units='A')
    z_bias.constant(t, a0*TpzShimEnd, units='A')
    offset.constant(t, (Offset + dTpzShim))
    
    return EvapTime_2
    
    
@stage
def dipole_evaporation_3(t):
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.customramp(t, EvapTime_3, ExpRamp, DipoleInt3, FinalDipole, dipoleEvapTau2, samplerate=1/step_dipole_evap)
    
    return EvapTime_3    
    
@stage
def hold(t):
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.constant(t, FinalDipole)
    
    return TimeHold
    
    
@stage
def TOF(t):
    # pulseblaster_0 outputs:
    dipole_switch.go_low(t)
    
    # ni_pci_6733_0 analog outputs:
    probe_repump.constant(t, 0.8)
       
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.constant(t, 0)
    dipole_split.constant(t, 0)
    
    # ni_usb_6343_1 analog outputs:
    x_bias.customramp(t, TimeTOF, LineRamp, TpxShimOffset, TpxShim, samplerate=1/step_TOF, units='A')
    y_bias.customramp(t, TimeTOF, LineRamp, TpyShimOffset, TpyShim, samplerate=1/step_TOF, units='A')
    z_bias.customramp(t, TimeTOF, LineRamp, a0*TpzShim, a0*TpzShimImg, samplerate=1/step_TOF, units='A')
    offset.constant(t, 0)
    
    return TimeTOF
    

@stage
def TOF_open_shutter(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_low(t)
    AOM_probe_1.go_low(t)
    shutter_z.go_high(t)
    
    # ni_pci_6733_0 outputs:
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, TpxShim, units='A')
    y_bias.constant(t, TpyShim, units='A')
    
    return 4*ms
    
    
@stage    
def repump_1(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_high(t)
    #camera_trigger_1.go_low(t)
    shutter_top_repump.go_high(t)
    
    # ni_pci_6733_0 outputs:
    MOT_repump.constant(t, 0.8)
    probe_repump.constant(t, 0.8)
    
    return TimeRepumpPulse
    

@stage    
def image_1(t):
    # pulseblaster_0 outputs:
    AOM_probe_1.go_high(t)
    YZ_1_Flea3.expose('shot1', t-30e-6, 'atoms')
    
    return TimeImage
    
@stage  
def download_image_1(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_low(t)
    #camera_trigger_1.go_high(t)
    AOM_probe_1.go_low(t)
    
    # ni_pci_6733_0 outputs:
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    probe_1.constant(t, 0)
    
    return TimeDownloadImage
    

@stage    
def repump_2(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_high(t)
    #camera_trigger_1.go_low(t)

    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t, 0.8)
    probe_repump.constant(t, 0.8)
    probe_1.constant(t, ProbeInt)
    
    return TimeRepumpPulse
    

@stage    
def image_2(t):
    # pulseblaster_0 outputs:
    AOM_probe_1.go_high(t)
    YZ_1_Flea3.expose('shot2', t-30e-6, 'probe')    
    return TimeImage
    

@stage  
def download_image_2(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_low(t)
    #camera_trigger_1.go_high(t)
    AOM_probe_1.go_low(t)
    shutter_z.go_low(t)
    shutter_top_repump.go_low(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    probe_1.constant(t, 0)
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 0, units='A')
    y_bias.constant(t, 0, units='A')
    z_bias.constant(t, 0, units='A')
    
    return TimeDownloadImage
    

@stage    
def repump_3(t):
    # pulseblaster_0 outputs:
    AOM_MOT_repump.go_high(t)
    #camera_trigger_1.go_low(t)
    dipole_switch.go_high(t)

    # ni_pci_6733_0 analog outputs:
    probe_repump.constant(t, 0.8)
    
    return TimeRepumpPulse
    

@stage    
def image_3(t):
    # Is a dark image, no light pulse:
    YZ_1_Flea3.expose('shot3', t-30e-6, 'dark')
    return TimeImage
    
   
@stage  
def download_image_3(t):
    # pulseblaster_0 outputs:
    #camera_trigger_1.go_high(t)
    AOM_MOT_cooling.go_high(t)
    AOM_probe_1.go_high(t)
    dipole_switch.go_low(t)
    
    return TimeDownloadImage

    
@stage
def off(t):
    # pulseblaster_0 outputs:
    UV_LED.go_high(t)
    dipole_switch.go_high(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t,  UVMOTRepump)
    MOT_cooling.constant(t, UVMOTAmplitude)
    probe_1.constant(t,     ProbeInt)
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, UVMOTCurrent, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 2.11*UVxShim, units='A')
    y_bias.constant(t,      UVyShim, units='A')
    z_bias.constant(t, -0.8*UVzShim, units='A')
    
    # ni_pci_6733_0 analog outputs:
    dipole_intensity.constant(t, DipoleInt)
    dipole_split.constant(t, DipoleSplit)
    
    # novatechdds9m_0 channel outputs:
    MOT_lock.setfreq(t,  (ResFreq + UVMOTFreq)*MHz)
    RF_evap.setfreq(t,  30*MHz)
    
    return 0.03
    
    
start()
t = 0
# transport_current_1_AI.acquire("testacquire1", 0, 10)
# transport_current_2_AI.acquire("testacquire2", 0, 10)
# transport_current_3_AI.acquire("testacquire3", 0, 10)
# transport_current_4_AI.acquire("testacquire4", 0, 10)
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
t += set_novatech(t)
t += decompress_2(t)
t += dipole_evaporation_1(t)
t += dipole_evaporation_2(t)
t += dipole_evaporation_3(t)
t += hold(t)
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

