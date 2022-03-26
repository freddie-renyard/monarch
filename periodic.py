from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space, plot_histogram
from math import cos, sin

# Define a simple ODE with sine and cosine functions.
ode = (
    lambda x, y : -y + cos(2 * x) - 0.05 * x,
    lambda x, y : x + sin(2 * y) - 0.05 * y
)

phase_space = PhaseSpace(
    ode_system = ode,
    resolution = 64,
    max_limit = 4,
    four_quadrant = True
)

plot_histogram(phase_space)
plot_2d_phase_space(phase_space, "a periodic function")