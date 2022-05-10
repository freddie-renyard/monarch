from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space, plot_histogram, plot_2d_simulation
from math import cos, sin
from monarch.uart import UART
from matplotlib import pyplot as plt

# Define a simple ODE with sine and cosine functions.
ode = (
    lambda x, y : -y + cos(2 * x) - 0.05 * x,
    lambda x, y : x + sin(2 * y) - 0.05 * y
)

phase_space = PhaseSpace(
    ode_system = ode,
    resolution = 64,
    max_limit = 4,
    dt = 0.001,
    four_quadrant = True
)

plot_2d_phase_space(phase_space, show_fig=True)
exit()
fpga = UART()

output_data = fpga.primary_eval(timesteps=10000)
print(output_data)
plot_2d_simulation(output_data, phase_space)