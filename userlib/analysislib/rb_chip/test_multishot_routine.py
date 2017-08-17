from pylab import *
import lyse

df = lyse.data()

finaldipole = df['FinalDipole']
x = df['test_singleshot_routine', 'x']

plot(finaldipole, x, 'bo')

show()