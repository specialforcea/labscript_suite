from __future__ import division
from lyse import *
from traces import *
import pandas as pd
import numpy as np

float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

class single_wave(object):

    """ Groups some methods for single-wave processing 
        in multi-shot analysis as done with lyse """
    
    def __init__(self, wave_name=None, single_shot_routine=None):
        self.df = data()
        self.single_shot_routine = str(single_shot_routine)
        self.wave_name = str(wave_name)
        self.wave = {}
        if self.wave_name == 'RunNo':
            self.wave_data = np.array([0.0])
            self.wave = ({'wave_method': self.single_shot_routine,
                          'wave_name': self.wave_name,                             
                          'wave_data': self.wave_data,
                          'wave_size': 0.0,
                          'wave_min': 0.0, 
                          'wave_max': 0.0,
                          'wave_uncer' : 0.0})
        else:
            if self.single_shot_routine != 'None':             
                self.wave_data = self.df[self.single_shot_routine, self.wave_name].values
            else:
                self.wave_data = self.df[self.wave_name].values    
            self.wave = ({'wave_method': self.single_shot_routine,
                          'wave_name': self.wave_name,                             
                          'wave_data': np.array(self.wave_data),
                          'wave_size': np.amax(self.wave_data.shape),
                          'wave_min': np.amin(self.wave_data), 
                          'wave_max': np.amax(self.wave_data),
                          'wave_uncer' : np.nanstd(self.wave_data)/np.sqrt(np.amax(self.wave_data.shape))})   

    def get_single(self, average=False):
        if average:
            print np.mean(self.wave['wave_data']), np.std(self.wave['wave_data'])
        return self.wave

    def log_wave(self):
        input = self.wave['wave_data']
        w_log = np.log(input)
        wave_log = ({'wave_method': 'None',
                'wave_name': 'log' + self.wave['wave_name'],                             
                'wave_data':np.array(w_log),
                'wave_size': np.amax(w_log.shape),
                'wave_min': np.amin(w_log), 
                'wave_max': np.amax(w_log),
                'wave_uncer' : 0.0})
        return wave_log
        
    def normalize(self, norm='max', val=1.0):
        input = self.wave['wave_data']
        if norm == 'max':
            w_norm = input/np.amax(input)
        elif norm == 'fraction':
            w_norm = input/(np.amin(input)+np.amax(input))
        elif norm == 'sum':
            w_norm = input/np.sum(input)
        elif norm == 'val':
            w_norm = input/val
        wave_nrm = ({'wave_method': 'None',
                'wave_name': 'norm_' + self.wave['wave_name'],                             
                'wave_data':np.array(w_norm),
                'wave_size': np.amax(w_norm.shape),
                'wave_min': np.amin(w_norm), 
                'wave_max': np.amax(w_norm),
                'wave_uncer' : 0.0})
        return wave_nrm
        
    def psd(self):
        input = self.wave['wave_data']
        w_psd = np.abs(np.fft.fftshift(np.fft.fft(input)))**2
        wave_psd = ({'wave_method': 'None',
                'wave_name': 'PSD_' + self.wave['wave_name'],                             
                'wave_data':np.array(w_psd),
                'wave_size': np.amax(w_psd.shape),
                'wave_min': np.amin(w_psd), 
                'wave_max': np.amax(w_psd),
                'wave_uncer' : 0.0})
        return wave_psd

    def reciprocal_space(self):
        input = self.wave['wave_data']
        w_reciprocal = np.reciprocal(input)
        wave_reciprocal = ({'wave_method': 'None',
                'wave_name': 'k_' + self.wave['wave_name'],                             
                'wave_data':np.array(w_reciprocal),
                'wave_size': np.amax(w_reciprocal.shape),
                'wave_min': np.amin(w_reciprocal), 
                'wave_max': np.amax(w_reciprocal),
                'wave_uncer' : 0.0})
        return wave_reciprocal

            
class multi_wave(object):
    
    """ Groups some methods for multi-wave processing 
        in multi-shot analysis as done with lyse """
        
    def __init__(self, *waves):    
        self.waves = waves
        df_dict = {}
        #for wave in self.waves:
        #    df_dict[wave['wave_name']] = wave['wave_data'].flatten()
        #self.sub_dataframe = pd.DataFrame.from_dict(df_dict, orient='columns') 
    
    def build_RunNo_wave(self):
        # Discriminate
        sizes = []
        for wave in self.waves:
            sizes.append(wave['wave_size'])
        big_wave_size = int(np.amax(sizes))
        for wave in self.waves:
            if wave['wave_size'] < big_wave_size:
                wave['wave_data'] = np.linspace(1, big_wave_size, big_wave_size)
                wave['wave_size'] = big_wave_size
                wave['wave_max'] = big_wave_size
                return wave
    
    def make_equal_size(self):
        """ Do max size by default """
        resized_waves = self.waves
        sizes = []
        for wave in resized_waves:
            sizes.append(wave['wave_size'])
        target_size = int(np.amax(sizes))
        for wave in resized_waves:
            if wave['wave_size'] < target_size:        
                wave['wave_data'] = np.zeros(target_size)
                wave['wave_size'] = target_size
        return resized_waves
        
    def wave_PSD(self):
        """ Returns power spectral density wave """
        x_input = self.waves[0]['wave_data']
        y_input = self.waves[1]['wave_data']
        x_tilde_arr = (2*np.pi*np.linspace(1/np.amax(x_input), 
                   1/np.amin(x_input[x_input!=0.]), 
                   np.amax(x_input.shape))) 
        mag_PSD =  np.fft.ifft(np.abs(np.fft.fftshift(np.fft.fft(y_input)))**2)
        wave_x_tilde = ({'wave_method':'None',
                        'wave_name': 'x_tilde',                             
                        'wave_data': x_tilde_arr,
                        'wave_size': np.amax(x_tilde_arr.shape),
                        'wave_min': np.amin(x_tilde_arr), 
                        'wave_max': np.amax(x_tilde_arr),
                        'wave_uncer' : 0.0})
        PSD = ({'wave_method': 'None',
                'wave_name': 'PSD',                             
                'wave_data':np.array(mag_PSD),
                'wave_size': np.amax(mag_PSD.shape),
                'wave_min': np.amin(mag_PSD), 
                'wave_max': np.amax(mag_PSD),
                'wave_uncer' : 0.0})
        return wave_x_tilde, PSD

    def average_over_wave(self, as_x=None, as_y=None, common=None):
        """ This method averages over wave from a dataframe.
        Inputs: as_x: associated x_wave
                as_y: associated y_wave (to be averaged)
                common:    common wave to index
        Outputs: x_wave: output x_wave
                 y_wave: output (averaged) y_wave   """        
        #as_x = self.wave
        pivoted_df = self.wavesub_dataframe.pivot(index=as_x['wave_name'], columns=common['wave_name'])
        averaged_df = pivoted_df[as_y['wave_name']].mean(axis=1)
        return averaged_df.index.values, averaged_df.values
        
    def fit_wave(self, trace='linear'):
        """ Returns fit of input wave using the traces.py wrapper """
        y_input = self.waves[1]['wave_data']
        x_input = np.reshape(self.waves[0]['wave_data'], y_input.shape)
        # Call fitting method with listed waves
        str_command_a = 'fit_'+trace+'('+str(x_input.tolist())+', '+str(y_input.tolist())+')'
        #print str_command_a
        fitted_parms, d_fitted_parms = eval(str_command_a)
        #print "Fit parameters for", trace, "method"
        #print fitted_parms
        x_space = np.linspace(np.amin(x_input), np.amax(x_input), 2**9)
        parstr = ''
        for par in fitted_parms:
            parstr = parstr + str(par) + ','
        str_command_b = trace+'('+str(x_space.tolist())+', '+parstr[:-1]+')'
        #print str_command_b
        x_fit_space, evaluated_fit = np.array(x_space), eval(str_command_b)
        x_fit_wave = ({'wave_method': 'multi_wave_fit',
              'wave_name': self.waves[0]['wave_name'],                             
              'wave_data': x_fit_space,
              'wave_size': self.waves[0]['wave_size'],
              'wave_min': self.waves[0]['wave_min'], 
              'wave_max': self.waves[0]['wave_max'],
              'wave_uncer' : self.waves[0]['wave_uncer']})
        eval_fit_wave = ({'wave_method': 'multi_wave_fit',
              'wave_fit_params': np.array(fitted_parms),
              'wave_name': (self.waves[1]['wave_name']+' '+trace+' fit'),                             
              'wave_data': np.array(evaluated_fit),
              'wave_size': np.amax(evaluated_fit),
              'wave_min': np.amin(evaluated_fit), 
              'wave_max': np.amax(evaluated_fit),
              'wave_uncer' : np.nanstd(evaluated_fit)/np.sqrt(np.amax(evaluated_fit.shape))})   
        return x_fit_wave, eval_fit_wave 

    def add_waves(self):
        a_input = self.waves[1]['wave_data']
        b_input = np.reshape(self.waves[0]['wave_data'], a_input.shape)
        added = a_input+b_input
        sum = ({'wave_method': 'None',
                'wave_name': self.waves[1]['wave_name']+'+'+self.waves[0]['wave_name'],                             
                'wave_data':np.array(added),
                'wave_size': np.amax(added.shape),
                'wave_min': np.amin(added), 
                'wave_max': np.amax(added),
                'wave_uncer' : 0.0})
        return sum
        
    def take_ratio(self):
        a_input = self.waves[1]['wave_data']
        b_input = np.reshape(self.waves[0]['wave_data'], a_input.shape)
        a_over_b = a_input/b_input
        ratio = ({'wave_method': 'None',
                'wave_name': self.waves[1]['wave_name']+'/'+self.waves[0]['wave_name'],                             
                'wave_data':np.array(a_over_b),
                'wave_size': np.amax(a_over_b.shape),
                'wave_min': np.amin(a_over_b), 
                'wave_max': np.amax(a_over_b),
                'wave_uncer' : 0.0})
        return ratio
        