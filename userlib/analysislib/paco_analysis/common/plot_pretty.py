from __future__ import division
from matplotlib import pyplot as plt

class plot_pretty(object):
    
    """Wraps plot line objects to overwrite their Line2D attributes"""
    
    def __init__(self, **attrs):
        self.attrs = attrs
        # Split marker attributes from line attributes
        self.marker_attrs = {}
        self.line_attrs = {}
        for key in self.attrs.keys():
            if 'marker' in str(key):
                self.marker_attrs[key] = self.attrs[key]
            else:
                self.line_attrs[key] = self.attrs[key]
            
    def __call__(self, basic_plot_function):
        def wrapped_plot_function(*args, **kwargs):
            lines, errorbar, is_fit = basic_plot_function(*args, **kwargs)
            if is_fit:
                plt.setp(lines, **self.line_attrs)
                plt.legend(handles=lines, loc=2,  prop={'size':10})
            elif errorbar:
				# Currently setting errorbar line and plot marker only, no caps
                plt.setp(lines[0], **self.marker_attrs)
                plt.setp(lines[2], **self.line_attrs)   
            else:
				# Manually turning linewidth off...probably fix this later..
                self.attrs['linewidth'] = 0.0
                plt.setp(lines, **self.attrs) 
                plt.legend(handles=lines, loc=2, prop={'size':10})  
            plt.tight_layout()
        return wrapped_plot_function