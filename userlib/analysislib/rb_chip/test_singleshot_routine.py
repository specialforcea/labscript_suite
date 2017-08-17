from pylab import *
import lyse

def calculate_x(run, df):
    print 'FinalDipole is', df['FinalDipole']
    x = df['FinalDipole']**2
    u_x = x/10.0
    print 'x =', x, '+/-', u_x
    run.save_result('x', x)
    run.save_result('u_x', u_x)

    
try:
    path
except NameError:
    import sys
    path = sys.argv[1]
    
run = lyse.Run(path)
df = lyse.data(path)

calculate_x(run, df)

