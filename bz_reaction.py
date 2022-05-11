from re import A
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_simulation
from matplotlib import pyplot as plt

a = 10.0
b = 5.0

ode = (
    lambda x, y: a - x - (4 * x * y) / (1 + x**2),
    lambda x, y: b * x * (1 - y / (1 + x**2))
)

phase_space = PhaseSpace(
    ode_system = ode,
    dt = 0.001,
    resolution = 128,
    max_limit = 16,
    four_quadrant = False,
    compress_space = False
)

test_data = phase_space.run_simulation([0,0], timesteps=40000)
plot_2d_simulation(test_data, None, phase_space)

plt.plot(test_data[:,0])
plt.show()