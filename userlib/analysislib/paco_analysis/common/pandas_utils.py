import inspect
import pandas
from numpy import *
from uncertainties import ufloat

def seriesify(*args):
    """Given a list of variables, packs them into a pandas Series

    Does some black magic, jumping out to the calling scope and looking
    up the names of the variables there to use them as the index to
    the Series

    You can *only* pass bare variables!
      seriesify(foo, u_foo, baz) is fine seriesify(3.2, sqrt(foo),
      not_a_variable) is all wrong!
    """
    frame = inspect.currentframe()
    upper_locals = frame.f_back.f_locals
    upper_names = {id(val): name for name, val in upper_locals.items()}
    names_dict = { upper_names[id(arg)]: nan_to_num(arg) for arg in args }
    names_list = [ upper_names[id(arg)] for arg in args ]
    return pandas.Series(names_dict, index=names_list)
    
def tuplify(df, name):
    """Pads a heirarchical column name with enough empty strings to be used as
    a column specification for a heirarchical DataFrame. For example:
        tuplify(df, ('roi_atoms', 'Fz')) = ('roi_atoms', 'Fz', '', '')
    """
    m = df.columns.nlevels
    if type(name) != tuple:
        name = (name,)
    while len(name) < m:
        name += ('',)
    return name

pandas.DataFrame.tuplify = tuplify

def maptuplify(df, names):
    return [tuplify(df, name) for name in names]
    
pandas.DataFrame.maptuplify = maptuplify
    
def u_global(name, prefix='u_', suffix=''):
    """Returns the uncertainty corresponding to a column. For example, the
    column ('roi_atoms', 'Fz', '', '') has uncertainties in the column
    ('roi_atoms', 'u_Fz', '', '')
    """
    if type(name) == tuple:
        i = len(name)-1
        while not name[i]:
            i -= 1
        t = name[:i]
        t += (prefix + name[i] + suffix,)
        t += name[i+1:]
        return t
    elif type(name) == str:
        return prefix + name + suffix
    else:
        return None
        
def multicols(df, names, how='all'):
    """Returns list of columns in a DataFrame which contain the substring name,
    even for heirarchical columns, for example:
        multicols(df, 'group A')
        [('group A', 'result 1'), ('group A', 'result 2')]
    """
    def colsearch(col, names, how):
        if how == 'all':
            h = all
        else:
            h = any
        if type(names) == str:
            names = [names]
        return h([any([name in subcol for subcol in col]) for name in names])
    return [col for col in df.columns if colsearch(col, names, how)]
    
pandas.DataFrame.multicols = multicols
    
def ufloatify(df, name):
    # Get the columns from the DataFrame containing name
    data = df[df.multicols(name)].values
    ydim, xdim = data.shape
    # Convert to ufloats pairwise
    data = array([map(lambda x: ufloat(*x), split(row, xdim/2)) for row in data])
    return data

pandas.ufloatify = ufloatify
    
def df_to_h5(df, hdf5_object, dset_name, strip_names=True, save_index=False):
    """Saves a pandas DataFrame object to a compound dataset (table) 
    named dset_name in hdf5_object: a hdf5 file or group, with named columns.
    
    strip_names : Keep the last non-empty element of the heirarchical
                  column title, e.g. ('roi_fit','Fz','','') yields the
                  column title 'Fz' in the table.
                  
    save_index : Save the index as named column(s).
    """
    # convert the DataFrame columns into a column name  
    def column_name(column):
        if type(column) == tuple and strip_names:
            i = 0
            column_name = ''
            while column[i]:
                column_name = column[i]
                i += 1
        else:
            column_name = str(column)
        return column_name

    # construct a numpy.dtype to create a compound dataset from the columns
    dset_dtype = []
    if save_index:
        index_names = df.index.names
        index_array = []
        for i, index_name in enumerate(index_names):
            index_list = df.index.get_level_values(index_name)
            if 'time' in index_list.dtype.name:
                # this is better but get_level_values has munged the conversion to datetime64
                # index_list = array(map(str, index_list))
                # ...so instead do this for now: 
                index_list = array(map(lambda t: t.strftime('%Y%m%dT%H%M%S'), zip(*df.index.values)[i]))
            dset_dtype.append((index_name, index_list.dtype.str))
            index_array.append(index_list)
    else:
        dset_dtype = []
    dset_dtype.extend((column_name(col), df[col].dtype.name) for col in df.columns)
    dset_table = empty(len(df), dtype=dset_dtype)
    # fill the compound array with the DataFrame and create the h5 dataset
    for col_name, col in zip(map(column_name, df.columns), df.columns):
        dset_table[col_name] = df[col].values
    if save_index:
        for index_name, index_list in zip(index_names, index_array):
            dset_table[index_name] = index_list
    hdf5_object.create_dataset(dset_name, data=dset_table)

pandas.DataFrame.to_h5 = df_to_h5
