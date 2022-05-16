from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_simulation

# The 2D form of the Oregonator reaction model
# of the Belusov Zhabontinsky (BZ) reaction.

epsilon = 10.0 ** -2
q = 10.0 ** -4
f = 0.0001

ode = (
    lambda x, y: x * (1 - x) + f * ((q - x) / (q  + x)) * y,
    lambda x, y: x - y
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 128,
    max_limit = 32.0,
    four_quadrant = False,
    compress_space = False
)

test_data = phase_space.run_simulation([30,16], timesteps=20000)
print(test_data)

plot_2d_simulation(test_data, None, phase_space, name="BZ Reaction: 2D Oregonator Model Simulation ")