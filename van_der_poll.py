from cmath import phase
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space, plot_2d_simulation, plot_reconstructed_space
from monarch.uart import UART

from matplotlib import pyplot as plt

# Define the 2D Van der Poll oscillator
mu = 1.0

ode = (
    lambda x, y: mu * (x - (1.0/3.0)*x**3 - y),
    lambda x, y: (1.0 / mu) * x
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 64,
    max_limit = 4,
    four_quadrant = True,
    compress_space=False
)

fpga = UART(phase_space)
output_data = fpga.primary_eval(timesteps=10000)
test_data = phase_space.run_simulation([2.0,2.0], 10000) 

plt.plot(output_data[:,0], output_data[:,1])
plt.plot(test_data[:,0], test_data[:,1])
plt.show()

plot_2d_simulation(output_data, test_data, phase_space)