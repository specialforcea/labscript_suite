# Gets parameters from a fit, puts them in a dataframe, prints them pretty.

from StringIO import StringIO
import pandas as pd
import numpy as np
import prettytable  


def get_params(fit_params, u_param=None):
    afit_params = np.asarray(fit_params)
    rfit_params = np.around(afit_params, decimals=6)
    ffit_params = rfit_params.tolist()
    if u_param is not None:
        au_param = np.asarray(u_param)
        ru_param = np.around(au_param, decimals=6)
        fu_param = ru_param.tolist()
    # Include uncertainties in fit parameters
        par_count = 0
        columns = []
        for par in ffit_params:
            par_count = par_count + 1
            columns.append('parameter_' + str(par_count))
        index = ['fit_parameters', 'u_parameter']
        df = pd.DataFrame(columns=columns, index=index)    
        df.loc[index[0]] = ffit_params
        df.loc[index[1]] = u_param
        output = StringIO()
        df.to_csv(output)
        output.seek(0)
        pt = prettytable.from_csv(output)
        print pt
    else:
    # No uncertainties in fit parameters
        par_count = 0
        columns = []
        for par in ffit_params:
            par_count = par_count + 1
            columns.append('parameter_' + str(par_count))
        index = ['fit_parameters']
        df = pd.DataFrame(columns=columns, index=index)    
        df.loc[index[0]] = ffit_params
        output = StringIO()
        df.to_csv(output)
        output.seek(0)
        pt = prettytable.from_csv(output)
        print  pt
        
