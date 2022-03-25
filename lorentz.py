from unittest import result
from monarch.monarch_objects import PhaseSpace

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
    resolution = 32,
    max_limit = 32,
    four_quadrant = True
)