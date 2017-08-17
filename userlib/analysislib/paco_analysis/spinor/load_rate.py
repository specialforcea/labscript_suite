from __future__ import division
from lyse import *
from pylab import *
from analysislib.common.traces import fit_exp, expRise

if not spinning_top:
    # if the script isn't being run from the lyse GUI, 
    # load the DataFrame from lyse and use the first shot
    try:
        df = data()
        path = df['filepath'][0]
    except:
        # if lyse isn't running, use an explicitly specified shot file path
        try:
            path = sys.argv[1]
        except:
            path = '20140409T163121_crossed_beam_bec_00.h5'
   
dataset = data(path)
run = Run(path)

show_plots = True
fit_success = False

try:
    print '\nRunning %s' % os.path.basename(__file__)
    t, y = run.get_trace('MOT_load')
    t_raw = t
    
    # COLLECTION EFFICIENCY
    eta = 0.1                                                           # collection efficiency of MOT fluorescence due to imperfect alignment, etc.

    # CONSTANTS
    h = 6.63e-34
    c = 3e8
    lambda_RbD2 = 780e-9
    Ephoton = h*c/lambda_RbD2

    # PHOTODETECTION
    responsivity = 0.5                                                  # A/W
    dBmList = arange(0, 80, 10)                                         # dB - 8 gain settings on the PD-36A
    gainList = 1.5e3*10**(dBmList/20)                                   # gain in V/A for PD-36A
    gain = gainList[dBmList == 60][0]                                   # V/A

    # FLUORESCENCE CAPTURE
    lens_distance = 125                                                 # distance between MOT and lens in mm - 5 bolt rows
    lens_aperture = 20                                                  # clear apeture of lens (diameter)
    solid_angle = (pi*(lens_aperture/2)**2)/(4*pi*lens_distance**2)     # solid angle subtended by collection optic
    gamma_per_V = (eta*solid_angle*gain*responsivity*Ephoton)**(-1)     # scattering rate/V

    # SATURATION PARAMETER
    background_V = 4/1.5*0.283                                          # background signal (V) with all beams on but no atoms                                    
    # background_V = 0.441                                              # background signal (V) with all beams on but no atoms - Navatar lens 20130719                                    
    # beam_power = mean([4.0, 4.2, 4.3, 4.4, 4.5])                      # total power in single MOT beam (at time of above calibration)
    beam_power = mean([4.12, 5.03, 3.50, 4.10, 2.80, 2.80])             # total power in single MOT beam (at time of above calibration) - 20130719
    power_cal = 6*beam_power/background_V                               # total MOT power in mW per background fluoro V
    MOT_power = y[1]*power_cal                                          # total MOT power in mW (assuming acqusition and MOT load begin simultaneously)
    beam_diameter = 0.8                                                 # MOT beam diameter in cm
    I_0 = MOT_power/(0.25*pi*beam_diameter**2)                          # MOT intensity in mW/cm^2 - uniform profile
    I_sat = 1.68                                                        # saturation intensity in mW/cm^2
    saturation = I_0/I_sat                                              # saturation parameter 

    # ATOMIC SCATTERING
    MHz = 1e6
    natural_linewidth = 2*pi*6.066*MHz                                  # rad/s
    MOT_detuning = 2*pi*dataset['MOT_detuning']                         # rad/s
    MOT_delta = MOT_detuning/(natural_linewidth/2)                      # detuning in half-linewidths
    gamma_per_atom = (natural_linewidth/2)*saturation/(1 + saturation + MOT_delta**2) # scattering rate per atom

    # OFFSET TIME ARRAY
    t = t[1:] - t[0]
    y = y[1:]
    
    # ATOMS PER VOLT CALIBRATION
    atoms_per_V = gamma_per_V/gamma_per_atom                            # atoms/V
    atoms = atoms_per_V*(y - y[0])
    
    # WINDOW MOT LOAD PERIOD
    load_filter = t <= dataset['MOT_load_time']
    t_load = t[load_filter]
    y_load = y[load_filter]
    atoms_load = atoms[load_filter]

    # RAW ATOM NUMBER CALCULATION (average the last 50ms of the MOT load)
    end_filter = t_load[-1] - t_load < 50e-3
    Natoms = mean(atoms_load[end_filter])

    # RECAPTURE ANALYSIS
    try:
        if dataset['MT_recapture']:
            time_globals = ['MOT_load_time', 'Zeeman_MOT_hold_time', 'MOT_bias_ramp_time']
            t_MOT_1 = sum([dataset[duration] for duration in time_globals])
            t_MOT_2 = t_MOT_1 + dataset['MOT_hold_time'] - 0.1*dataset['MOT_hold_time']
            MOT_filter = (t < t_MOT_2) * (t > t_MOT_2 - 50e-3)
            N_MOT = mean(atoms[MOT_filter])
            time_globals = ['CMOT_time', 'molasses_time', 'MT_hold_time']
            t_recapture_1 = t_MOT_2 + 0.1*dataset['MOT_hold_time'] + sum([dataset[duration] for duration in time_globals]) + 30e-3
            t_recapture_2 = t_recapture_1 + dataset['MT_recapture_time'] - 30e-3
            recapture_filter = (t > t_recapture_1) * (t < t_recapture_1 + 50e-3)
            N_recapture = mean(atoms[recapture_filter])
            recapture_fraction = N_recapture/N_MOT
            recapture = True
        else:
            recapture = False
    except KeyError:
        recapture = False
    
    # FITTING
    try:
        (final, decay_time), (d_final, d_decay_time) = fit_exp(t_load, atoms_load, fix_initial=True)
        fit_success = True
    except Exception as e:
        print 'Fit failed: %s' % str(e)

    # LOAD RATE (use the first second of data)
    start_filter = t_load < 1
    rate = mean(atoms_load[start_filter])/mean(t_load[start_filter])

    # SAVE RESULTS
    run.save_results('MOT_load_rate', rate, 'MOT_number', Natoms, 'MOT_power', MOT_power)
    run.save_results('MOT_decay', decay_time, 'u_MOT_decay', d_decay_time)
    if recapture:
        run.save_results('recapture_fraction', recapture_fraction)
    

    # PLOTTING
    if show_plots:
        fig = figure('load rate')
        ax = fig.add_subplot(111)
        l1 = ax.plot(t, atoms, 'b-', label='atoms')
        if fit_success:
            l2 = ax.plot(t_load, expRise(t_load, 0, final, decay_time), 'r-', linewidth=3, label='atoms (fit)')
        ax.plot(t_load[start_filter], rate*t_load[start_filter], 'g-', linewidth=3)
        ax.plot(t_load[end_filter], Natoms + 0*t_load[end_filter], 'g-', linewidth=3)
        if recapture:
            ax.plot(t[MOT_filter], N_MOT + 0*t[MOT_filter], 'g-', linewidth=3)
            ax.plot(t[recapture_filter], N_recapture + 0*t[recapture_filter], 'g-', linewidth=3)
        ax.set_ylabel('atoms')
        ax.set_ylim(ymin=0)
        ax.set_xlabel('time (s)')
        
        # plot raw voltage on secondary y-axis
        ax2 = ax.twinx()
        l3 = ax2.plot(t, y, color = '0.75', alpha=0, label='voltage')
        ax2.set_ylim(array(ax.get_ylim())/atoms_per_V+y[1])
        ax2.set_ylabel('voltage')
        
        # make legend for multi-axis plot
        if fit_success:
            lines = l1+l2
            labels = [li.get_label() for li in lines]
            ax.legend(lines, labels, loc='upper left')

        # title and grid
        plot_title = 'initial load rate = %0.3e atoms/s' % rate
        if recapture:
            plot_title += '\n' + 'recapture fraction = %.2f' % (100*recapture_fraction) + "%"
        title(plot_title)
        grid(True)
        show()

        # ATOM NUMBER BANNER
        fig2 = figure('atom number', figsize=(15,3), facecolor='w')
        str = r'$N$ = %.3e' % Natoms
        text(0.5, 0.7, str, ha='center', va='top',fontsize=130)
        gca().axison = False
        show()
        
        # RECAPTURE BANNER
        if recapture:
            fig3 = figure('recapture fraction', figsize=(15,3), facecolor='w')
            str = r'$N_{\mathrm{MT}} / N_{\mathrm{MOT}}$ = %.2f' % (100*recapture_fraction) + "%"
            text(0.5, 0.7, str, ha='center', va='top', fontsize=120)
            gca().axison = False
            show()


except:
    print 'No MOT_load trace in ' + path