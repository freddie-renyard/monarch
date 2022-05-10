from cmath import phase
from matplotlib import pyplot as plt
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_components, plot_2d_phase_space, plot_histogram, plot_2d_simulation, plot_reconstructed_space
from monarch.uart import UART

from matplotlib import pyplot as plt
import numpy as np

# A model of a calcium-induced calcium release model in cells, which
# accurately models calcium ion oscillations in the cytosol of cells.

# Information for this model is taken from Chapter 2 of Hamid Soleimani's 
# PhD thesis, which can be found on ArKIV: https://arxiv.org/pdf/2108.04928.pdf

# Declaration of the model's Hill coefficients
n = 2.0
m = 2.0
p = 2.0

# Declaration of model parameters
v_m_2 = 100.0
v_m_3 = 700.0
k_2 = 1.0
k_r = 15.0
k_a = 2.5
k_f = 0
k = 8.0

beta = 1.0

# Declaration of the biochemical rate equations, which
# form parts of the final model.
z_0 = 1.0
z_1 = 6.0
z_2 = lambda x    : v_m_2 * ((x ** n) / (k_2 ** n + x ** n))
z_3 = lambda x, y : v_m_3 * ((y ** m) / (k_r ** m + y ** m)) * ((x ** p) / (k_a ** p + x ** p)) 

# Declaration of the full ODE system
# First equation is dx / dt, second is dy / dt
ode = (
    lambda x, y : z_0 - z_2(x) + z_3(x,y) + k_f * y - k * x + z_1 * beta,
    lambda x, y : z_2(x) - z_3(x,y) - k_f * y
)

phase_space = PhaseSpace(
    ode_system = ode,
    resolution = 60,
    max_limit = 32, 
    dt = 0.001,
    four_quadrant = False,
    verbose = False
)

plot_histogram(phase_space)
plot_2d_phase_space(phase_space)
plot_2d_components(phase_space)

fpga = UART()
output_state = fpga.primary_eval(timesteps=2000)

plot_2d_simulation(output_state, phase_space)