from time import time
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_simulation, plot_histogram, plot_2d_phase_space, plot_2d_components
from monarch.uart import UART
from matplotlib import pyplot as plt

# An compilation of the FitzHugh-Nagumo (FHN) neuron, a simplification of the 
# Hodkin-Huxley dynamic neuron model.

# Information for this model is taken from Chapter 2 of Hamid Soleimani's 
# PhD thesis, which can be found on ArKIV: https://arxiv.org/pdf/2108.04928.pdf

I = 0.9
a = 1

ode = (
    lambda x, y : x - (x ** 3) / 3 - y + I,
    lambda x, y : a * (x + 0.7 - 0.8 * y)
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 128,
    max_limit = 4,
    four_quadrant = True,
    compress_space= False
)

fpga = UART(phase_space)
output_data = fpga.primary_eval(timesteps=30000)
#test_data = phase_space.run_simulation([0, 0], timesteps=50000)

plot_2d_simulation(output_data, None, phase_space, name="FitzHugh-Nagumo (FHN) Neuron Model Simulation")