#####################################################################
#                                                                   #
# /NovaTechDDS9M.py                                                 #
#                                                                   #
# Copyright 2017, JQI                                               #
#                                                                   #
# This file is part of the module labscript_devices, in the         #
# labscript suite (see http://labscriptsuite.org), and is           #
# licensed under the Simplified BSD License. See the license.txt    #
# file in the root of the project for the full license.             #
#                                                                   #
#####################################################################
from labscript_devices import runviewer_parser, labscript_device, BLACS_tab, BLACS_worker

from labscript import IntermediateDevice, AnalogQuantity, config, LabscriptError, set_passed_properties, MHz
from labscript_utils.unitconversions import NovaTechDDS9mFreqConversion

import numpy as np
import labscript_utils.h5_lock, h5py
import labscript_utils.properties

        
@labscript_device
class ChipFPGA(IntermediateDevice):
    description = 'ChipFPGA'
    allowed_children = [AnalogQuantity]
    clock_limit = 50*MHz # confirmed

    MAX_FREQ = 50e6 # confirmed
    
    NUM_WIRES = 96
    NUM_WIRE_GROUPS = 6
    
    MEMORY_SIZE_BYTES = 512*1024 # 512 KB, confirmed
    
    def __init__(self, name, parent_device, visa_resource="ASRL7::INSTR", **kwargs):

        IntermediateDevice.__init__(self, name, parent_device, **kwargs)
        self.BLACS_connection = visa_resource
        self.wireamps = [] #wireamps is a list of AnalogQuantity objects?
        for i in range(self.NUM_WIRES):
            self.wireamps.append(AnalogQuantity("%s_wire%02damp" % (self.name, i), self, 'wire%02damp' % i))
            
        self.groupphases = []
        for i in range(self.NUM_WIRE_GROUPS):
            self.groupphases.append(AnalogQuantity("%s_group%02dphase" % (self.name, i), self, 'group%02dphase' % i))
            
        self.groupfreqs = []
        for i in range(self.NUM_WIRE_GROUPS):
            self.groupfreqs.append(AnalogQuantity("%s_group%02dfreq" % (self.name, i), self, 'group%02dfreq' % i,
                                   unit_conversion_class=NovaTechDDS9mFreqConversion))
                                   
        self.control_bytes = {}
        self.visa_resource = visa_resource

    def set_control_byte(self, name, value):
        self.control_bytes[name] = np.uint8(value)

    def setamp(self, t, wire, value, *args, **kwargs):
        self.wireamps[wire].constant(t, value, *args, **kwargs)
        
    def setfreq(self, t, group, value, *args, **kwargs):
        self.groupfreqs[group].constant(t, value, *args, **kwargs)
        
    def setphase(self, t, group, value, *args, **kwargs):
        self.groupphases[group].constant(t, value, *args, **kwargs)
        
    def quantise_freq(self, data, device):   #could replace quantise_functions with ones to take care of register values. save register
        if not isinstance(data, np.ndarray):  # values to h5 files directly. (original data maybe necessary too)
            data = np.array(data)             # at this moment, I can't think of a case where we need different frequency for different
        # Ensure that frequencies are within bounds:      # group of wires.
        if np.any(data > self.MAX_FREQ)  or np.any(data < 0):   #accuracy of frequency is from F_ref/(2**16-1) to F_ref, depends on N&C)
            raise LabscriptError('%s %s '%(device.description, device.name) +
                              'can only have frequencies between 0 and {}MHz, '.format(self.MAX_FREQ/1e6) + 
                              'the limit imposed by %s.'%self.name)
                              
        scale_factor = (2**16-1)/self.MAX_FREQ # number of bits to be confirmed
        # It's faster to add 0.5 then typecast than to round to integers first:
        data = np.array((scale_factor*data)+0.5,dtype=np.uint16)
        return data, scale_factor
        
    def quantise_phase(self, data, device):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # ensure that phase wraps around:
        data %= 360
        scale_factor = 255/360.
        # It's faster to add 0.5 then typecast than to round to integers first:
        data = np.array((scale_factor*data)+0.5,dtype=np.uint8)
        return data, scale_factor
        
    def quantise_amp(self,data,device):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # ensure that amplitudes are within bounds:
        if np.any(data > 5 )  or np.any(data < 0):
            raise LabscriptError('%s %s '%(device.description, device.name) +
                              'can only have amplitudes between 0 and 5 (Volts peak to peak approx), ' + 
                              'the limit imposed by %s.'%self.name)
        scale_factor = 255/5.0
        # It's faster to add 0.5 then typecast than to round to integers first:
        data = np.array(scale_factor*data + 0.5,dtype=np.uint8)
        
        return data, scale_factor
        
    def generate_code(self, hdf5_file):     
        memory_per_instruction = self.NUM_WIRES # 1 byte per amplitude
        memory_per_instruction += self.NUM_WIRE_GROUPS*1 # 1 byte per phase
        memory_per_instruction += self.NUM_WIRE_GROUPS*2 # 2 bytes per frequency: to be confirmed
        max_instructions = int(self.MEMORY_SIZE_BYTES/memory_per_instruction)
        
        # All the devices share a clock, so we can check just one of them to see whether the number of instructions will fit in RAM
        num_instructions = len(self.wireamps[0].raw_output)
        if num_instructions > max_instructions:
            raise LabscriptError('%s can only support %d instructions. ' % (self.name, max_instructions) +
                                 'Please decrease the sample rates of devices on the same clock, ' + 
                                 'or connect %s to a different pseudoclock.' % self.name)
                                 
        quantised_freqs = np.zeros((num_instructions, self.NUM_WIRE_GROUPS), dtype=np.uint16) # datatype to be confirmed
        quantised_amps = np.zeros((num_instructions, self.NUM_WIRES), dtype=np.uint8)
        quantised_phases = np.zeros((num_instructions, self.NUM_WIRE_GROUPS), dtype=np.uint8)
        
        for i in range(self.NUM_WIRES):
            output = self.wireamps[i]
            quantised_amp, amp_scale_factor = self.quantise_amp(output.raw_output, output)# how is raw_output generated
            quantised_amps[:, i] = quantised_amp
            
        for i in range(self.NUM_WIRE_GROUPS):
            output = self.groupfreqs[i]
            quantised_freq, freq_scale_factor = self.quantise_freq(output.raw_output, output)
            quantised_freqs[:, i] = quantised_freq
            
            output = self.groupphases[i]
            quantised_phase, phase_scale_factor = self.quantise_phase(output.raw_output, output)
            quantised_phases[:, i] = quantised_phase
        
        grp = self.init_device_group(hdf5_file)
        grp.create_dataset('amplitude_table', compression=config.compression, data=quantised_amps)
        grp.create_dataset('frequency_table', compression=config.compression, data=quantised_freqs) 
        grp.create_dataset('phase_table', compression=config.compression, data=quantised_phases)
        self.set_property('frequency_scale_factor', freq_scale_factor, location='device_properties')
        self.set_property('amplitude_scale_factor', amp_scale_factor, location='device_properties')
        self.set_property('phase_scale_factor', phase_scale_factor, location='device_properties')
        control_bytes_group = grp.create_group('control_bytes')
        for key, value in self.control_bytes.items():
            control_bytes_group.attrs[key] = value
            
        


import time

from blacs.tab_base_classes import Worker, define_state
from blacs.tab_base_classes import MODE_MANUAL, MODE_TRANSITION_TO_BUFFERED, MODE_TRANSITION_TO_MANUAL, MODE_BUFFERED  
from blacs.device_base_class import DeviceTab

from qtutils import UiLoader
import os
import chipfpgaui
import numpy as np
from PyQt4 import QtGui
import visa


@BLACS_tab
class ChipFPGATab(DeviceTab):
    def initialise_GUI(self):
        # Capabilities
        num = {'freq_div':6,'ph_delay':6,'wireamp':96}
        base_units       =  {'freq':'MHz', 'amp':'V',   'phase': 'Degrees'}
        base_min         =  {'freq':0,     'amp':0,     'phase': 0}
        base_max         =  {'freq':50,    'amp':1,     'phase': 360}
        base_step        =  {'freq':0.01,  'amp':0.02,  'phase': 0.01}
        base_decimals    =  {'freq':3,     'amp':3,     'phase': 3}
        
        digital_properties = {'usb_channel':{'base_unit':'Arb','min':0,'max':255,'step':1,'decimals':3}}
        analog_properties = {'ao1':{'base_unit':'Arb','min':0,'max':255,'step':1,'decimals':3}}
        dds_properties = {'dds1':{'base_unit':'Arb','min':0,'max':255,'step':1,'decimals':3}}
        
        self.col_num = num['freq_div'] + num['ph_delay'] +num['wireamp']
        self.create_digital_outputs(digital_properties)
        self.create_analog_outputs(analog_properties)
        self.create_dds_outputs(dds_properties)
        dds_widgets,ao_widgets,do_widgets = self.auto_create_widgets()

        
        
        #self.auto_place_widgets(ao_widgets,do_widgets)
        
        self.ui = UiLoader().load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'chipfpga.ui'))        
        self.get_tab_layout().addWidget(self.ui)

        # Create and set the primary worker
        # self.create_worker("main_worker",ChipFPGAWorker)
        # self.primary_worker = "main_worker"

        # Set the capabilities of this device
        self.supports_smart_programming(True) 

        # Connect signals for buttons
        self.ui.Load_Button.clicked.connect(self.load)
        self.ui.Write_Button.clicked.connect(self.write)
        self.ui.Read_button.clicked.connect(self.read)
        self.ui.correct_button.clicked.connect(self.correct)

        #initialize the status to display
        self.read_status = []
        self.write_status = []

        self.rm = visa.ResourceManager()
        self.chipfpga_usb = self.rm.open_resource("ASRL7::INSTR")

    def numto2chr(self,number):
        num = format(number,'016b')
        num1 = num[8:16]
        num2 = num[0:8]
        num1_int = int(num1,2)
        num2_int = int(num2,2)
        num1_str = chr(num1_int)
        num2_str = chr(num2_int)
        return num1_str,num2_str
    
    def read_buffer(self,number):
        import timeit
        md = self.chipfpga_usb
        string_in_buffer = ''
        start_time = timeit.default_timer()
        while True:
            a = md.visalib.read(md.session,md.bytes_in_buffer)[0]
            if len(string_in_buffer) == number :
                break
            else:
                string_in_buffer = string_in_buffer + a
                
            if(timeit.default_timer() - start_time > 5):
                return 'TIMEOUT'
        return string_in_buffer
    def load(self):
        table_dir = str(self.ui.file_dir_Edit.text())
        table = np.load(table_dir)
        num_row, num_col = table.shape
        self.ui.load_table.setRowCount(num_row)
        self.ui.load_table.setColumnCount(num_col)
		
	    
        
        for i in range(num_row):
            for j in range(num_col):
                a = str(table[i,j])
                self.ui.load_table.setItem(i,j, QtGui.QTableWidgetItem(a))
        return table
    def write_raw(self,string):
        l = len(string)
        num1_str,num2_str = self.numto2chr(l)
        md = self.chipfpga_usb
        md.write_raw('\x01')
        if self.read_buffer(1) != '\x00':
            return 'ERROR'
        md.write_raw(num1_str)
        if self.read_buffer(1) != '\x01':
            return 'ERROR'
        md.write_raw(num2_str)
        if self.read_buffer(1) != '\x02':
            return 'ERROR'    
        for i in range(l):
            md.write_raw(string[i])
            if self.read_buffer(1) != '\x03':
                return 'ERROR'
    def write(self):
        
        num_row = self.ui.load_table.rowCount()
        num_col = self.ui.load_table.columnCount()
        table = np.zeros((num_row,num_col),dtype = int)
        #self.ui.write_status_edit.setText(str(self.ui.load_table.item(1,1).text()))
        

        for i in range(num_row):
            for j in range(num_col):
                
                table[i,j] = int(self.ui.load_table.item(i,j).text())

        load_list = table.reshape((1,-1))
        chr_load_list = map(chr,load_list[0,:])
        # load_string = str(chr_load_list)

        self.write_raw(chr_load_list)
        
        read_table = self.read_table_func(num_row*num_col)
        # self.ui.read_status_edit.setText(str(load_string.shape))
        
        
        compare = read_table == table
        wrong_ind = np.where(compare==0)
        self.ui.read_status_edit.setText(str(type(wrong_ind[0])))
        wrong_col = wrong_ind[0]
        wrong_num = len(wrong_col)
        for w in range(wrong_num):
            row_ind = wrong_ind[0][w]
            col_ind = wrong_ind[1][w]
            address = row_ind *self.col_num + col_ind
            value = table[row_ind,col_ind]
            self.change_one(self.chipfpga_usb,address,value)
            
        read_table = self.read_table_func(num_row*num_col)
        if (read_table == table).all() == False:
            self.ui.read_status_edit.setText('mismatch')

        else:
            self.ui.read_status_edit.setText('success, data in SRAM match table')


        

        #self.ui.write_status_edit.setText(str(uni_load_list))
  
    def read_table_func(self,number):
        read_string = self.read_raw(number)
        read_list = map(ord,read_string)
        read_array = np.array(read_list)
        read_table = read_array.reshape((-1,self.col_num))
        return read_table
    def read_raw(self,number):
        md = self.chipfpga_usb
        num1_str,num2_str = self.numto2chr(number)
        md.write_raw('\x04')
        a = self.read_buffer(1)
        if  a == '\x00':
            md.write_raw(num1_str)
            b = self.read_buffer(1)
            if b == '\x04':
                md.write_raw(num2_str)
            else:
                return '\x04',b
        else:
            return '\x00',a
        
        return self.read_buffer(number)
    def read(self):
        
        number = int(self.ui.byte_to_read_edit.text())
        read_str = self.read_raw(self.col_num*number)

        rlist = map(ord,read_str)
        table = np.array(rlist)
        table_r = table.reshape((-1,self.col_num))
        table_l = self.load()
		
        num_row, num_col = table_r.shape
        self.ui.read_table.setRowCount(num_row)
        self.ui.read_table.setColumnCount(num_col)
        
        for i in range(num_row):
            for j in range(num_col):
                a = str(table_r[i,j])
                self.ui.read_table.setItem(i,j, QtGui.QTableWidgetItem(a))


        # num_row = self.ui.load_table.rowCount()
        # num_col = self.ui.load_table.columnCount()
        # table_w = np.zeros((num_row,num_col),dtype = int)
        # #self.ui.write_status_edit.setText(str(self.ui.load_table.item(1,1).text()))


        # for i in range(num_row):
        #     for j in range(num_col):
                
        #         table_w[i,j] = int(self.ui.load_table.item(i,j).text())

        if (table_r == table_l).all() == False:

            self.ui.write_status_edit.setText('mismatch')
        else:

            self.ui.write_status_edit.setText('match')

        return table_r
    def correct(self):
        md = self.chipfpga_usb
        address = int(self.ui.correct_byte_edit.text())
        value = int(self.ui.correct_value_edit.text())
        self.change_one(md,address,value)

    def change_one(self,md,address,value):
        num1_str,num2_str = self.numto2chr(address)
        md.write_raw('\x07')
        a = self.read_buffer(1)
        if  a != '\x00':
            return '\x00',a
        md.write_raw('\x01')
        a = self.read_buffer(1)
        if  a != '\x07':
            return '\x07',a
        md.write_raw(num1_str)
        a = self.read_buffer(1)
        if  a != '\x08':
            return '\x08',a
            
        md.write_raw(num2_str)
        a = self.read_buffer(1)
        if  a != '\x09':
            return '\x09',a
        md.write_raw(chr(value))
        a = self.read_buffer(1)
        if  a != '\x0a':
            return '\x0a',a
@BLACS_worker
class ChipFPGAWorker(Worker):
    def init(self):
        global serial; import serial
        global h5py; import labscript_utils.h5_lock, h5py
        self.rm = visa.ResourceManager()
        self.chipfpga_usb = self.rm.open_resource("ASRL7::INSTR")
		self.col_num = 108
		self.smart_cache = []

    def check_remote_values(self):
        # Get the currently output values:
        pass
    def read_buffer(self,number):
        import timeit
        md = self.chipfpga_usb
        string_in_buffer = ''
        start_time = timeit.default_timer()
        while True:
            a = md.visalib.read(md.session,md.bytes_in_buffer)[0]
            if len(string_in_buffer) == number :
                break
            else:
                string_in_buffer = string_in_buffer + a
                
            if(timeit.default_timer() - start_time > 5):
                return 'TIMEOUT'
        return string_in_buffer   
	def numto2chr(self,number):
        num = format(number,'016b')
        num1 = num[8:16]
        num2 = num[0:8]
        num1_int = int(num1,2)
        num2_int = int(num2,2)
        num1_str = chr(num1_int)
        num2_str = chr(num2_int)
        return num1_str,num2_str
    
	def write_raw(self,string):
        l = len(string)
        num1_str,num2_str = self.numto2chr(l)
        md = self.chipfpga_usb
        md.write_raw('\x01')
        if self.read_buffer(1) != '\x00':
            raise Exception('suppose to get '\x00' but not')
        md.write_raw(num1_str)
        if self.read_buffer(1) != '\x01':
            raise Exception('suppose to get '\x01' but not')
        md.write_raw(num2_str)
        if self.read_buffer(1) != '\x02':
            raise Exception('suppose to get '\x02' but not')    
        for i in range(l):
            md.write_raw(string[i])
            if self.read_buffer(1) != '\x03':
                raise Exception('suppose to get '\x03' but not')    
    def write(self, table_data):
        
        
        num_col = self.num_col
		num_row = len(table_data)/num_col
        
      

        load_list = table_data.reshape((1,-1))
        chr_load_list = map(chr,load_list[0,:])
        # load_string = str(chr_load_list)

        self.write_raw(chr_load_list)
        
        
        # self.ui.read_status_edit.setText(str(load_string.shape))
        
        while (True):
            read_table = self.read_table_func(num_row*num_col)
            compare = read_table == table_data
            wrong_ind = np.where(compare==0)
            
            wrong_col = wrong_ind[0]
            wrong_num = len(wrong_col)
            for w in range(wrong_num):
                row_ind = wrong_ind[0][w]
                col_ind = wrong_ind[1][w]
                address = row_ind *self.col_num + col_ind
                value = table_data[row_ind,col_ind]
                self.change_one(self.chipfpga_usb,address,value)
                
            read_table = self.read_table_func(num_row*num_col)
            if (read_table == table_data).all() == False:
                

            else:
                break


        

        #self.ui.write_status_edit.setText(str(uni_load_list))
	def change_one(self,md,address,value):
        num1_str,num2_str = self.numto2chr(address)
        md.write_raw('\x07')
        a = self.read_buffer(1)
        if  a != '\x00':
            return '\x00',a
        md.write_raw('\x01')
        a = self.read_buffer(1)
        if  a != '\x07':
            return '\x07',a
        md.write_raw(num1_str)
        a = self.read_buffer(1)
        if  a != '\x08':
            return '\x08',a
            
        md.write_raw(num2_str)
        a = self.read_buffer(1)
        if  a != '\x09':
            return '\x09',a
        md.write_raw(chr(value))
        a = self.read_buffer(1)
        if  a != '\x0a':
            return '\x0a',a
    def read_table_func(self,number):
        read_string = self.read_raw(number)
        read_list = map(ord,read_string)
        read_array = np.array(read_list)
        read_table = read_array.reshape((-1,self.col_num))
        return read_table
    def read_raw(self,number):
        md = self.chipfpga_usb
        num1_str,num2_str = self.numto2chr(number)
        md.write_raw('\x04')
        a = self.read_buffer(1)
        if  a == '\x00':
            md.write_raw(num1_str)
            b = self.read_buffer(1)
            if b == '\x04':
                md.write_raw(num2_str)
            else:
                return '\x04',b
        else:
            return '\x00',a
        
        return self.read_buffer(number)
    def transition_to_buffered(self,device_name,h5file,initial_values,fresh):


        # Pretty please reset your memory pointer to zero:

        # Store the initial values in case we have to abort and restore them:
        self.initial_values = initial_values
        # Store the final values to for use during transition_to_static:
        self.final_values = np.zeros((1,self.num_col),dtype = int)
        
        with h5py.File(h5file) as hdf5_file:
            group = hdf5_file['/devices/' + device_name]
            
            if 'TABLE_DATA' in group:
                table_data = group['TABLE_DATA'][:]

        write(table_data)

                # Save these values into final_values so the GUI can
                # be updated at the end of the run to reflect them:
        self.smart_cache = table_data 
			
		t_d = table_data.reshape((-1,self.num_col))
		self.final_values = t_d[t_d.shape[0]-1]

        # Now program the buffered outputs:
        
            # Store the table for future smart programming comparisons:
            
            # new table is longer than old table
               

            # Get the final values of table mode so that the GUI can
            # reflect them after the run:
            

            # Transition to table mode:
            
            
                # Transition to hardware synchronous updates:
                
                # We are now waiting for a rising edge to trigger the output
                # of the second table pair (first of the experiment)
           
                # Output will now be updated on falling edges.
                


        return self.final_values

    def abort_transition_to_buffered(self):
        return self.transition_to_manual(True)

    def abort_buffered(self):
        # TODO: untested
        return self.transition_to_manual(True)

    def transition_to_manual(self,abort = False):
        
        pass
        return True

    def shutdown(self):

        # return to the default baud rate

        self.chipfpga_usb.close()


    


        

        












