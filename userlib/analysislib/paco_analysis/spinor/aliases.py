from analysislib.common.pandas_utils import u_global

# Imaging mode
orientation = 'side'
ODi = 'OD'
# ODi = 'OD0'
fit_function = 'Gaussian'
# fit_function = 'ThomasFermi'
# if orientation == 'side':
    # prefix = 'u_'
    # suffix = ''
# else:
    # prefix = ''
    # suffix = '_Uncert'

# Raw atom numbers
n_atoms_fluoro = ('load_rate', 'MOT_number', '', '')
n_atoms = (orientation, 'absorption', 'OD', 'atom_number')
n_atoms = ('roi_atoms', 'Nt', '', '')
# n_atoms_fit = ('roi_atoms', 'Nt_fit', '', '')
n_atoms_fit = ('side', 'absorption', 'OD', 'Gaussian_Nint')

# Analysis results
load_rate = ('load_rate', 'MOT_load_rate', '', '')

# Common globals
tDrop = ('drop_time', '', '', '')
t_drop = tDrop

# Image fit parameters
try:
    n_atoms_fit
except:
    n_atoms_fit = (orientation, 'absorption', ODi, fit_function + '_Nint')
OD_max = (orientation, 'absorption', ODi, fit_function + '_Amp')
widthX = (orientation, 'absorption', ODi, fit_function + '_XW')
widthY = (orientation, 'absorption', ODi, fit_function + '_YW')
positionX = (orientation, 'absorption', ODi, fit_function + '_X0')
positionY = (orientation, 'absorption', ODi, fit_function + '_Y0')

# Image fit parameter uncertainties
u_n_atoms_fit = u_global(n_atoms_fit)
u_widthY = u_global(widthY)
u_widthX = u_global(widthX)
u_positionX = u_global(positionX)
u_positionY = u_global(positionY)

# OD_max_top = ('top', 'absorption', ODi, fit_function + '_Amp')
# widthX_top = ('top', 'absorption', ODi, fit_function + '_XW')
# widthY_top = ('top', 'absorption', ODi, fit_function + '_YW')
# u_widthX_top = ('top', 'absorption', ODi, 'u_' + fit_function + '_XW')
# u_widthY_top = ('top', 'absorption', ODi, 'u_' + fit_function + '_YW')
# positionX_top = ('top', 'absorption', ODi, fit_function + '_X0')
# positionY_top = ('top', 'absorption', ODi, fit_function + '_Y0')
# u_positionX_top = ('top', 'absorption', ODi, 'u_' + fit_function + '_X0')
# u_positionY_top = ('top', 'absorption', ODi, 'u_' + fit_function + '_Y0')

# Image attributes
capture_roi_offset_X = (orientation, 'OffsetX', '', '') # relative to the camera chip
capture_roi_offset_Y = (orientation, 'OffsetY', '', '') # relative to the camera chip
fit_roi_offset_X = (orientation, 'absorption', 'OD', 'Fit_fit_offset_X') # relative to the capture ROI
fit_roi_offset_Y = (orientation, 'absorption', 'OD', 'Fit_fit_offset_Y') # relative to the capture ROI

# Scattering cross-section
sigma0 = 3*(780e-9)**2/(2*3.14)

# Constants
red = (0.67,0,0)
blue = (0,0,0.67)
green = (0,0.67,0)
