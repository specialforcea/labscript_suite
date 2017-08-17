"""
Project   : python BEC analysis
Filename  : test_plot

Created on  : 2014 Dec 14 17:24
Author      : aputra
"""

import matplotlib.pyplot as plt
from pylab import *
import PhysConstants as phc

x = linspace(0, 5, 10)
y = x ** 2

# figure()
# plot(x, y, 'r')
# xlabel('x')
# ylabel('y')
# title('title')
# show()

fig = plt.figure(1)
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes # left, bottom, width, height (range 0 to 1)
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# main figure
axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

# insert
axes2.plot(y, x, 'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title')

# show()


# Update the matplotlib configuration parameters:
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots()

ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$y$')
ax.set_title('title')

show()

print phc.hbar