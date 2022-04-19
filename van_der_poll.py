from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space

# Define the 2D Van der Poll oscillator
mu = 5.0

ode = (
    lambda x, y: mu * (x - (1.0/3.0)* x**3 - y),
    lambda x, y: (1.0 / mu) * x
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 64,
    max_limit = 8,
    four_quadrant = True
)

plot_2d_phase_space(phase_space)