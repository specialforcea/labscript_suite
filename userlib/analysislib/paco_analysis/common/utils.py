from __future__ import division
#from formatuncertainty import *
from uncertainties import ufloat

si = {'s':1, 'ms':1e-3, 'us':1e-6, 'ns':1e-9, 
      'Hz':1, 'kHz':1e3, 'MHz':1e6, 'GHz':1e9,
      'm':1, 'cm':1e-2, 'mm':1e-3, 'um':1e-6, 'nm':1e-9,
      'mG':1e-3, 'uG':1e-6}

def asdatetime(timestr):
    import pandas as pd
    # tz = localtz().zone
    tz = 'Australia/Melbourne'
    # tz = None
    return pd.Timestamp(timestr, tz=tz)
    
def format_unc(x, u_x):
    return '{:.1uS}'.format(ufloat(x, abs(u_x)))

def chunks(l, n):
    """ Yield successive chunks from l which are at least of length n
    From http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def udictify(*args):
    """Given a list of ufloats, packs them into a name: (val, u_val) dict

    Does some black magic, jumping out to the calling scope and looking
    up the names of the variables there to use them as dictionary keys.

    You can *only* pass bare variables!
      udictify(foo, u_foo, baz) is fine udictify(3.2, sqrt(foo),
      not_a_variable) is all wrong!
    """
    import inspect
    frame = inspect.currentframe()
    upper_locals = frame.f_back.f_locals
    upper_names = {id(val): name for name, val in upper_locals.items()}
    names_dict = { upper_names[id(arg)]: arg for arg in args }
    return names_dict
    
def errorplot(x, y, **kwargs):
    from pylab import errorbar
    if 'uncertainties' in str(type(x[0])):
        xval = [val.n for val in x]
        xerr = [val.s for val in x]
    else:
        xval = x
        xerr = None
    if 'uncertainties' in str(type(y[0])):
        yval = [val.n for val in y]
        yerr = [val.s for val in y]
    else:
        yval = y
        yerr = None
    errorbar(xval, yval, xerr=xerr, yerr=yerr, **kwargs)
