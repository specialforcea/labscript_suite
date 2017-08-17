from __future__ import division
from utils import *
from pandas_utils import *
import pandas as pd

def recent_sequences(df, n=1):
    # Return the n most recent sequences
    return df.ix[df.index.levels[0][-n:]]

def sequences(df, ts=None, before=None, after=None, last=None):
    if last is not None:
        # Return the n most recent sequences
        return recent_sequences(df, last)
    if ts is not None:
        # Return particular sequences specified by a single value or 
        # array of timestamp strings or Timestamps
        if type(ts) not in [list, ndarray, pandas.tseries.index.DatetimeIndex]:
            ts = [ts]
        ts = map(asdatetime, ts)
        # subdf =  df.xs(ts[0])
        # for t in ts[1:]:
        #    subdf.combine_first(df.xs(t))
        # return subdf
        return df[df.index.get_level_values('sequence').isin(ts)]
    else:
        # Filter between two times (which need not correspond to any index)
        return df.truncate(before=asdatetime(after), after=asdatetime(before))

def dropnans(df, subset=None):
    # Omit purely NaN columns 
    df = df.dropna(axis=1, how='all')
    if subset:
        # Check if all columns in subset have correct depth
        subset = df.maptuplify(subset)
        # Ensure that none of subset got deleted in the column cull above
        subset = [col for col in subset if col in df.columns]
    # Omit rows with NaNs in specified columns
    return df.dropna(subset=subset)
        
def sequence_strings(df):
    # Return a list of sequence strings
    if df.index.nlevels > 1:
        # sequence = df.index.levels[0] # this is broken in v0.12+
        # sequence = set(df.index.get_level_values(0)) # this is broken in v0.13.1+
        sequence = unique([df.index[i][0] for i in range(len(df))])
    else:
        sequence = [df['sequence'][0]]
    seq_str = [s.strftime('%Y%m%dT%H%M%S') for s in sequence]
    return seq_str
    
def sequence_string(df, line_length=3):
    # Return a comma separated string of sequences,
    # line breaking each line_length sequences.
    seq_str = sequence_strings(df)
    if line_length:
        return '\n'.join([', '.join(x) for x in chunks(seq_str, line_length)])
    else:
        return seq_str
        
def changing(df, name, minpts=3):
     return len(set(df[name])) == len(df) and len(df) > minpts

pd.DataFrame.recent_sequences = recent_sequences        
pd.DataFrame.sequences = sequences
pd.DataFrame.dropnans = dropnans
pd.DataFrame.sequence_strings = sequence_strings
pd.DataFrame.sequence_string = sequence_string
pd.DataFrame.changing = changing
