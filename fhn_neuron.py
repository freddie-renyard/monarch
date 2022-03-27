from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_histogram, plot_2d_phase_space, plot_2d_components

# An compilation of the FitzHugh-Nagumo (FHN) neuron, a simplification of the 
# Hodkin-Huxley dynamic neuron model.

# Information for this model is taken from Chapter 2 of Hamid Soleimani's 
# PhD thesis, which can be found on ArKIV: https://arxiv.org/pdf/2108.04928.pdf

I = 0.5
a = 1

ode = (
    lambda x, y : x - (x ** 3) / 3 - y + I,
    lambda x, y : a * (x + 0.7 - 0.8 * y)
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 64,
    max_limit = 8,
    four_quadrant = True
) 

plot_2d_components(phase_space)