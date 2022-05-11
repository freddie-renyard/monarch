from time import time
from unicodedata import name
from unittest import result

from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space, plot_3d_phase_space, plot_histogram
from monarch.uart import UART

from matplotlib import pyplot as plt
import numpy as np

# Define the Lorentz attractor.
sigma = 10
rho = 28
beta = 8/3

ode = (
	lambda x,y,z : sigma * (y - x),
	lambda x,y,z : x * (rho - z) - y,
	lambda x,y,z : x * y - beta * z
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 32,
    max_limit = 64,
    four_quadrant = True,
    compress_space = False
)

#plot_histogram(phase_space)

fpga = UART(phase_space)
output_data = fpga.primary_eval(timesteps=100) 
test_data = phase_space.run_simulation([2,2,2],timesteps=100)
print(output_data)

ax = plt.axes(projection='3d')
ax.plot3D(output_data[:, 0], output_data[:, 1], output_data[:, 2], 'gray')
ax.plot3D(test_data[:, 0], test_data[:, 1], test_data[:, 2], 'blue')
plt.show()