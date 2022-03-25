from unicodedata import name
from unittest import result

from matplotlib.pyplot import plot
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space, plot_3d_phase_space

from matplotlib import pyplot as plt
import numpy as np

# Define the Lorentz attractor.
sigma = 10
rho = 50
beta = 8/3

ode = (
	lambda x,y,z : sigma * (y - x),
	lambda x,y,z : x * (rho - z) - y,
	lambda x,y,z : x * y - beta * z
)

phase_space = PhaseSpace(
    ode_system = ode,
    resolution = 32,
    max_limit = 32,
    four_quadrant = True
)

plot_3d_phase_space(phase_space, name='Lorentz attractor')