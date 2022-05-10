from cmath import phase
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space, plot_2d_simulation, plot_reconstructed_space
from monarch.uart import UART

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
output_data = fpga.primary_eval(timesteps=50000)

print(output_data)

plot_2d_simulation(output_data, phase_space)