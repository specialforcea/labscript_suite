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
    AOM_optical_pumping.go_low(t)
    shutter_optical_pumping.go_low(t)
    AOM_probe_1.go_high(t)
    shutter_z.go_low(t)
    AOM_probe_2.go_low(t)
    shutter_x.go_low(t)
    UV_LED.go_low(t)
    camera_trigger_1.go_high(t)
    vacuum_shutter.go_low(t)
    RF_TTL.go_low(t)
    shutter_top_repump.go_low(t)
    dipole_switch.go_low(t)
    
    # ni_usb_6343_0 digital outputs:
    curent_supply_2_enable.go_high(t)
    
    # ni_usb_6343_1 digital outputs:
    kepco_enable_0.go_high(t)
    kepco_enable_1.go_high(t)
    
    # ni_pci_6733_0 analog outputs:
    MOT_repump.constant(t,  UVMOTRepump)
    MOT_cooling.constant(t, UVMOTAmplitude)
    probe_1.constant(t,     ProbeInt)
    
    # ni_pci_6733_1 analog outputs:
    microwave_attenuator.constant(t, 6) # Volts
    uwave_power_1.constant(t,        1) # Volt
    uwave_power_2.constant(t,        0) # Volt
    
    # ni_usb_6343_0 analog outputs:
    transport_current_2.constant(t, UVMOTCurrent, units='A')
    
    # ni_usb_6343_1 analog outputs:
    x_bias.constant(t, 2.11*UVxShim, units='A')
    y_bias.constant(t,      UVyShim, units='A')
    z_bias.constant(t, -0.8*UVzShim, units='A')
    offset.constant(t, 0)
    
    # novatechdds9m_0 channel outputs:
    MOT_lock.setfreq(t,  (ResFreq + UVMOTFreq)*MHz)
    MOT_lock.setamp(t,  0.7077) # 0 dBm
    MOT_lock.setphase(t,      0)
    RF_evap.setfreq(t,   30*MHz)
    RF_evap.setamp(t,   0)      # -80 dBm
    RF_evap.setphase(t,       0)
    # microwave.setfreq(110.06565*MHz)
    # microwave.setamp(1.0)    # +3 dBm
    # microwave.setphase(0)
    # dipole_AO.setfreq(100*MHz)
    # dipole_AO.setamp(0)      # -80 dBm
    # dipole_AO.setphase(0)
    
    # novatechdds9m_1 channel outputs:
    # uwave1.setfreq(t,    100*MHz)
    # uwave1.setamp(t,    0)      # -80 dBm
    # uwave1.setphase(t,        0)
    # uwave2.setfreq(t,    100*MHz)
    # uwave2.setamp(t,    0)      # -80 dBm
    # uwave2.setphase(t,        0)
    
    return prep_duration
    

@stage    
def UVMOT(t):
    # pulseblaster_0 outputs:
    UV_LED.go_high(t)

    # # ni_pci_6733_1 analog outputs:
    # uwave_power_1.constant(t, 0)   # Volt
     
    # # novatechdds9m_1 channel outputs:
    # uwave1.setfreq(t,    110.06565*MHz)
    # uwave1.setamp(t,    1.0)    # +3 dBm
    
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
    MOT_lock.frequency.customramp(t,  MOT_time, HalfGaussRamp, MOTFreq + ResFreq, CMOTFreq+ResFreq, CMOTCaptureWidth, samplerate=1/step_cMOT, units='MHz')
    RF_evap.setfreq(t,   100*MHz)
    RF_evap.setamp(t,   0)      # -80 dBm
     
    # # novatechdds9m_1 channel outputs:
    # uwave1.frequency.customramp(t, MOT_time, HalfGaussRamp, MOTFreq, CMOTFreq, CMOTCaptureWidth, samplerate=1/step_cMOT, units='MHz')
    # uwave1.setamp(t,    0.7077) # 0 dBm
    # uwave2.setfreq(t,    30*MHz)
    
    return MOT_time

    
@stage
def MOToff(t):    
    t0 = t
    # ni_usb_6343_0 analog outputs:
    t += transport_current_2.customramp(t, MOToff_time, HalfGaussRamp, CMOTCurrent, 0, CMOTCaptureWidth, samplerate=1/step_cMOT, units='A')
    return t - t0
    
    
@stage
def molasses(t):
    t0 = t
    shutter_optical_pumping.go_high(t)
    MOT_repump.constant(t, RepumpMol)
    transport_current_2.constant(t, MOTCurrent, units='A')
    x_bias.constant(t, 2.11*0.44, units='A')
    y_bias.constant(t, 0.42, units='A')
    z_bias.constant(t, -0.8*-0.04, units='A')
    curent_supply_2_enable.go_low(t)
    t += MOT_lock.frequency.customramp(t, TimeMol, ExpRamp, StartFreqMol + ResFreq , EndFreqMol + ResFreq , TauMol, samplerate=1/step_molasses, units='MHz')
    
    return t - t0
    
    
@stage
def optical_pump(t):
    t0 = t
    
    AOM_MOT_repump.go_low(t)
    AOM_MOT_cooling.go_low(t)
    shutter_MOT_cooling.go_low(t)

    MOT_repump.constant(t, RepumpOpPump)
    MOT_cooling.constant(t, 0)
    optical_pumping.constant(t, OpPumpAmplitude)
    
    x_bias.constant(t, 2.11*OpxShim, units='A')
    y_bias.constant(t, OpyShim, units='A')
    z_bias.constant(t, -0.8*OpzShim, units='A')
    MOT_lock.frequency.constant(t, ResFreq + EndFreqMol, units='MHz') # Redundant? Already set in previous stage.
    
    t += TimePrePump
    
    AOM_MOT_repump.go_high(t)
    AOM_optical_pumping.go_high(t)
    
    t += TimePump
    
    return t - t0
    
    
@stage
def magnetic_trap(t):
    AOM_MOT_repump.go_low(t)
    shutter_MOT_repump.go_low(t)
    AOM_optical_pumping.go_low(t)
    shutter_optical_pumping.go_low(t)
    AOM_probe_1.go_low(t)
    curent_supply_2_enable.go_high(t)
    curent_supply_3_enable.go_high(t)
    MOT_repump.constant(t, 0)
    optical_pumping.constant(t, 0)
    transport_current_2.customramp(t, TrapTime,  HalfGaussRamp, MOTCaptureCurrent, IM, CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    x_bias.customramp(t, TrapTime, HalfGaussRamp,  2.11*CapxShim, 2.11*0.44, CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    y_bias.customramp(t, TrapTime, HalfGaussRamp,  CapyShim, 0.42,           CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    z_bias.customramp(t, TrapTime, HalfGaussRamp, -0.8*CapzShim, -0.8*-0.04, CaptureWidth, samplerate=1/step_magnetic_trap, units='A')
    return TrapTime
    
    
@stage
def magnetic_transport(t):
    t0 = t
    AOM_MOT_repump.go_high(t)
    AOM_probe_1.go_high(t)
    curent_supply_4_enable.go_high(t)
    MOT_repump.constant(t, 0.8)
    transport_current_1.customramp(t, TimeMove01, Poly4Asymmetric, IPM, WidthPM, SkewPM, 1, Cf=[Cf5, AccelFrac, 0, Vel0], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove01, PolyHalf2, 0, IM, WidthM, SkewM,          Cf=[Cf5, AccelFrac, 0, Vel0], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove01, Poly4,     0, IMB, WidthMB, SkewMB,       Cf=[Cf5, AccelFrac, 0, Vel0], samplerate=1/step_magnetic_transport, units='A')                               
    x_bias.constant(t,  2.11*0.44, units='A')
    y_bias.constant(t,  0.42, units='A')
    z_bias.constant(t, -0.8*-0.04, units='A')                               
    
    t += TimeMove01

    curent_supply_1_select_line0.go_high(t)
    curent_supply_1_enable.go_high(t)
    transport_current_1.constant(t, 0, units='A')
    transport_current_2.customramp(t, TimeMove02, PolyHalf2, 1, IM, WidthM, SkewM,       Cf=[Cf2, d1, Vel0, Vel1], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove02, Poly4,     1, IMB, WidthMB, SkewMB,    Cf=[Cf2, d1, Vel0, Vel1], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove02, PolyExp,   0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel0, Vel1], samplerate=1/step_magnetic_transport, units='A')
    
    t += TimeMove02
    
    curent_supply_2_select_line0.go_high(t)
    transport_current_1.customramp(t, TimeMove03, PolyExp, 0, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel1, Vel2], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.constant(t, 0, units='A')
    transport_current_3.customramp(t, TimeMove03, Poly4,   2, IMB, WidthMB, SkewMB,    Cf=[Cf2, d1, Vel1, Vel2], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove03, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel1, Vel2], samplerate=1/step_magnetic_transport, units='A')

    t += TimeMove03
    
    curent_supply_3_select_line0.go_high(t)
    transport_current_1.customramp(t, TimeMove04, PolyExp, 1, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel2, Vel3], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove04, PolyExp, 0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel2, Vel3], samplerate=1/step_magnetic_transport, units='A')                      
    transport_current_3.constant(t, 0, units='A')
    transport_current_4.customramp(t, TimeMove04, PolyExp, 2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel2, Vel3], samplerate=1/step_magnetic_transport, units='A')

    t += TimeMove04 
    
    curent_supply_4_select_line0.go_high(t)
    transport_current_1.customramp(t, TimeMove05, PolyExp, 2, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel3, Vel4], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove05, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel3, Vel4], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove05, PolyExp, 0, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel3, Vel4], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.constant(t, 0, units='A')
    
    t += TimeMove05

    curent_supply_1_select_line0.go_low(t)
    curent_supply_1_select_line1.go_high(t)
    transport_current_1.constant(t, 0, units='A')
    transport_current_2.customramp(t, TimeMove06, PolyExp, 2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel4, Vel5], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove06, PolyExp, 1, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel4, Vel5], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove06, PolyExp, 0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel4, Vel5], samplerate=1/step_magnetic_transport, units='A')
    
    t += TimeMove06
    
    curent_supply_2_select_line0.go_low(t)
    curent_supply_2_select_line1.go_high(t)
    transport_current_1.customramp(t, TimeMove07, PolyExp, 0, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel5, Vel6], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.constant(t, 0, units='A')
    transport_current_3.customramp(t, TimeMove07, PolyExp, 2, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel5, Vel6], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove07, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel5, Vel6], samplerate=1/step_magnetic_transport, units='A')

    t += TimeMove07
 
    curent_supply_3_select_line0.go_low(t)
    curent_supply_3_select_line1.go_high(t)
    transport_current_1.customramp(t, TimeMove08, PolyExp, 1, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel6, Vel7], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove08, PolyExp, 0, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel6, Vel7], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.constant(t, 0, units='A')
    transport_current_4.customramp(t, TimeMove08, PolyExp, 2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel6, Vel7], samplerate=1/step_magnetic_transport, units='A')

    t += TimeMove08
                                   
    curent_supply_4_select_line0.go_low(t)
    curent_supply_4_enable.go_low(t)
    kepco_enable_0.go_low(t)
    transport_current_1.customramp(t, TimeMove09, PolyExp, 2, IB, WidthB, SkewB, ExpB, Cf=[Cf2, d1, Vel7, Vel8], samplerate=1/step_magnetic_transport, units='A')
    transport_current_2.customramp(t, TimeMove09, PolyExp, 1, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel7, Vel8], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove09, Poly4,   0, IBT, WidthBT, SkewBT,    Cf=[Cf2, d1, Vel7, Vel8], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.constant(t, -1, units='A')
    x_bias.customramp(t, TimeMove09, LineRamp,  2.11*0.44, TpzShim,     samplerate=1/step_magnetic_transport, units='A')
    y_bias.customramp(t, TimeMove09, LineRamp,  0.42, TpyShimInitial,   samplerate=1/step_magnetic_transport, units='A')
    z_bias.customramp(t, TimeMove09, LineRamp, -0.8*-0.04, a0*TpzShim,  samplerate=1/step_magnetic_transport, units='A')
    
    t += TimeMove09
    
    curent_supply_1_select_line1.go_low(t)
    curent_supply_1_enable.go_low(t)
    curent_supply_4_select_line0.go_high(t)
    curent_supply_4_select_line1.go_high(t)
    curent_supply_4_enable.go_high(t)
    transport_current_1.constant(t, -1, units='A')
    transport_current_2.customramp(t, TimeMove10, PolyExp,   2, IL, WidthL, SkewL, ExpL, Cf=[Cf2, d1, Vel8, Vel9], samplerate=1/step_magnetic_transport, units='A')
    transport_current_3.customramp(t, TimeMove10, Poly4,     1, IBT, WidthBT, SkewBT,    Cf=[Cf2, d1, Vel8, Vel9], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove10, PolyHalf1, 0, IFT, WidthFT, SkewFT,    Cf=[Cf2, d1, Vel8, Vel9], samplerate=1/step_magnetic_transport, units='A')
    x_bias.constant(t, TpxShim, units='A')
    y_bias.constant(t, TpyShimInitial, units='A')
    z_bias.constant(t, a0*TpzShim, units='A')
    
    t += TimeMove10
    
    curent_supply_2_select_line1.go_low(t)
    curent_supply_2_enable.go_low(t)
    transport_current_1.constant(t,  0, units='A')
    transport_current_2.constant(t, -1, units='A')
    transport_current_3.customramp(t, TimeMove11, Poly4,     2, IBT, WidthBT, SkewBT, Cf=[Cf2, d2, Vel9, 0], samplerate=1/step_magnetic_transport, units='A')
    transport_current_4.customramp(t, TimeMove11, PolyHalf1, 1, IFT, WidthFT, SkewFT, Cf=[Cf2, d2, Vel9, 0], samplerate=1/step_magnetic_transport, units='A')
       
    t += TimeMove11    
    
    dipole_intensity.constant(t, DipoleInt)
    dipole_split.constant(t, DipoleSplit)
    transport_current_3.constant(t, -1, units='A')
    transport_current_4.customramp(t, TimeMove12, LineRamp, IFT,            TopTrapCurrent, samplerate=1/step_magnetic_transport, units='A')
    y_bias.customramp(t,              TimeMove12, LineRamp, TpyShimInitial, TpyShimTrap,    samplerate=1/step_magnetic_transport, units='A')
    
    t += TimeMove12
    
    return t - t0
   
   
@stage
def get_ready_for_TOF(t):
    # If not doing evaporation, set the coil currents to
    # what they would be at the end of evaporation:
    transport_current_1.constant(t,  0, units='A')    
    transport_current_2.constant(t,  0, units='A')    
    transport_current_3.constant(t,  0, units='A')    
    transport_current_4.constant(t,  0, units='A')   
    curent_supply_3_select_line1.go_low(t)
    curent_supply_3_enable.go_low(t)    
    x_bias.constant(t, TpxShimEnd, units='A')
    y_bias.constant(t, TpyShimEnd, units='A')
    z_bias.constant(t, a0*TpzShimEnd, units='A')
    return 0
   
   
@stage
def evaporate(t):
    RF_TTL.go_high(t)
    dipole_switch.go_high(t)
    curent_supply_3_select_line1.go_low(t)
    curent_supply_3_enable.go_low(t)
    RF_mixer.constant(t, 0.6)
    transport_current_2.constant(t, 0, units='A')
    transport_current_3.constant(t, 0, units='A')
    transport_current_4.constant(t, TopTrapCurrent, units='A')
    y_bias.constant(t, TpyShimTrap, units='A')
    RF_evap.frequency.customramp(t, RFevapTime, LineRamp, StartRF, EndRF, samplerate=1/step_evap, units='MHz')
    RF_evap.setamp(t,   0.7077)      # 0 dBm
    return RFevapTime

    
@stage
def decomp_1(t):
    transport_current_4.customramp(t, DecompTime1, LineRamp, TopTrapCurrent, MiddleCurrent, samplerate=1/step_decomp_1, units='A')
    y_bias.customramp(t,              DecompTime1, LineRamp, TpyShimTrap,    TpyShimTrap1,  samplerate=1/step_decomp_1, units='A')
    RF_evap.frequency.customramp(t,   DecompTime1, LineRamp, EndRF,          EndFreq,       samplerate=1/step_decomp_1, units='MHz')
    return DecompTime1
   
   
@stage
def decomp_2(t):
    transport_current_4.customramp(t, DecompTime1, LineRamp, MiddleCurrent,  FinalCurrent,  samplerate=1/step_decomp_2, units='A')
    y_bias.customramp(t,              DecompTime1, LineRamp, TpyShimTrap1,   TpyShimTrap2,  samplerate=1/step_decomp_2, units='A')
    MOT_lock.setfreq(t, (ResFreq+ImageFreq2)*MHz)
    RF_evap.setfreq(t, EndRF*MHz)
    return DecompTime2
    
    
@stage
def dipole_evaporation(t):
    t0 = t
    RF_TTL.go_low(t)
    dipole_intensity.customramp(t,    EvapTime, ExpRamp, DipoleInt,    DipoleInt2,       dipoleEvapTau,  samplerate=1/step_dipole_evap)
    dipole_split.customramp(t,        EvapTime, ExpRamp, DipoleSplit,  FinalDipoleSplit, dipoleEvapTau,  samplerate=1/step_dipole_evap)
    RF_mixer.constant(t, 0)
    transport_current_4.customramp(t, EvapTime, ExpRamp, FinalCurrent, 0,                dipoleEvapTau,  samplerate=1/step_dipole_evap, units='A')
    y_bias.customramp(t,              EvapTime, ExpRamp, TpyShimTrap2, TpyShimTrap3,     dipoleEvapTau,  samplerate=1/step_dipole_evap, units='A')
    
    t += EvapTime
    
    dipole_intensity.customramp(t,    EvapTime2, ExpRamp, DipoleInt2,  DipoleInt3,       dipoleEvapTau2, samplerate=1/step_dipole_evap)  
    x_bias.constant(t, TpxShimEnd, units='A')
    y_bias.constant(t, TpyShimEnd, units='A')
    z_bias.constant(t, a0*TpzShimEnd, units='A')
    
    t +=EvapTime2
    
    dipole_intensity.customramp(t,    EvapTime3, ExpRamp, DipoleInt3,  FinalDipoleInt,   dipoleEvapTau3, samplerate=1/step_dipole_evap)
    
    t +=EvapTime3
    
    return t - t0
    
    
@stage
def TOF(t):
    dipole_switch.go_low(t)
    curent_supply_4_select_line0.go_low(t)
    curent_supply_4_select_line1.go_low(t)
    curent_supply_4_enable.go_low(t)
    probe_repump.constant(t, 0.8)
    dipole_intensity.constant(t, 0)
    dipole_split.constant(t, 0)
    z_bias.customramp(t, TimeTOF, LineRamp, a0*TpzShimEnd, a0*TpzShimImg, samplerate=1/step_TOF, units='A')
    MOT_lock.setfreq(t, (ResFreq+ImageFreq)*MHz)
    return TimeTOF

    
@stage
def TOF_open_shutter(t):
    t0 = t
    AOM_MOT_repump.go_low(t)
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    
    # t +=TOF_Open_Shutter1
    
    shutter_z.go_high(t)
    shutter_top_repump.go_high(t)
    
    #t += TOF_Open_Shutter2
    t += TOF_Open_Shutter
    
    return t - t0
    

@stage
def TOF_open_shutter_X(t):
    t0 = t
    AOM_MOT_repump.go_low(t)
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    
    # t +=TOF_Open_Shutter1
    
    shutter_x.go_high(t)
    shutter_top_repump.go_high(t)
    
    #t += TOF_Open_Shutter2
    t += TOF_Open_Shutter
    
    return t - t0
    
    
@stage
def take_image(t, name, frame_type, repump=True):
    t0 = t
    if repump:
        # Repump pulse:
        AOM_MOT_repump.go_high(t)
        MOT_repump.constant(t, 0.8)
        probe_repump.constant(t, 0.8)
        if frame_type != 'dark':
            probe_1.constant(t, ProbeInt)
        exposure_time = repumppulse + TimeImg
    else:
        exposure_time = TimeImg
    XY_1_Flea3.expose(name, t, frame_type, exposure_time)
    
    if repump:
        t += repumppulse
        
    if frame_type != 'dark':
        # Image pulse:
        AOM_probe_1.go_high(t)
        
    t += TimeImg
    
    # Download image
    if frame_type == 'dark':
        AOM_MOT_repump.go_high(t)
        AOM_probe_1.go_high(t)
    else:
        AOM_MOT_repump.go_low(t)
        AOM_probe_1.go_low(t)
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    probe_1.constant(t, 0)

    t += TimeDownloadImg
    
    return t - t0
    

@stage
def take_image_YZ(t, name, frame_type, repump=True):
    t0 = t
    if repump:
        # Repump pulse:
        AOM_MOT_repump.go_high(t)
        MOT_repump.constant(t, 0.8)
        probe_repump.constant(t, 0.8)
        if frame_type != 'dark':
            probe_1.constant(t, ProbeInt)
        exposure_time = repumppulse + TimeImg
    else:
        exposure_time = TimeImg
    YZ_1_Flea3.expose(name, t, frame_type, exposure_time)
    
    if repump:
        t += repumppulse
        
    if frame_type != 'dark':
        # Image pulse:
        AOM_probe_1.go_high(t)
        
    t += TimeImg
    
    # Download image
    if frame_type == 'dark':
        AOM_MOT_repump.go_high(t)
        AOM_probe_1.go_high(t)
    else:
        AOM_MOT_repump.go_low(t)
        AOM_probe_1.go_low(t)
    MOT_repump.constant(t, 0)
    probe_repump.constant(t, 0)
    probe_1.constant(t, 0)

    t += TimeDownloadImg
    
    return t - t0
    
    
def prepare_for_dark_frame(t):
    shutter_z.go_low(t)
    shutter_x.go_low(t)
    shutter_top_repump.go_low(t)
    x_bias.constant(t, 0, units='A')
    y_bias.constant(t, 0, units='A')
    z_bias.constant(t, 0, units='A')
    
    
@stage
def off(t):
    dipole_switch.go_high(t)
    UV_LED.go_high(t)
    MOT_repump.constant(t, RepumpMol)
    probe_repump.constant(t, 0)
    MOT_cooling.constant(t, 0.6)
    AOM_MOT_cooling.go_high(t)
    probe_1.constant(t, ProbeInt)
    dipole_intensity.constant(t, DipoleInt)
    dipole_split.constant(t, DipoleSplit)
    MOT_lock.setfreq(t, (ResFreq + UVMOTFreq)*MHz)
    RF_evap.setfreq(t, 30*MHz)
    RF_evap.setamp(t, 0) # -80 dBm
    return TimeOff
      
start()
t = 0

transport_current_1_AI.acquire('coil1_load', 0, 1)
transport_current_2_AI.acquire('coil2_load', 0, 1)
transport_current_3_AI.acquire('coil3_load', 0, 1)
transport_current_4_AI.acquire('coil4_load', 0, 1)

t += prep(t)
t += UVMOT(t)
t += cMOT(t)
if MOT_ONLY:
    t += MOToff(t)
    t += cooldown_time
else:
    t += molasses(t)
    t += optical_pump(t)
    t += magnetic_trap(t)
    t += magnetic_transport(t)
    if EVAPORATE:
        t += evaporate(t)
        t += decomp_1(t)
        t += decomp_2(t)
        t += dipole_evaporation(t)
    else:
        t += get_ready_for_TOF(t)
    t += TOF(t)
    t += TOF_open_shutter_X(t)
    t += take_image_YZ(t, 'absorption', 'atoms', repump=True)
    t += take_image_YZ(t, 'absorption', 'flat', repump=True)
    # This is done during downloading of previous frame so that shutters
    # are closed and fields are settled in time for the dark exposure:
    prepare_for_dark_frame(t-TimeDownloadImg)
    t += take_image_YZ(t, 'absorption', 'dark', repump=True)
    t += cooldown_time
    t += off(t)
stop(t+1*ms)
