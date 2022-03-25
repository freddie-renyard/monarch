from lib2to3.pygram import python_grammar_no_print_statement
from tkinter import Y
from monarch.monarch_objects import PhaseSpace
from monarch.visual_tools import plot_2d_phase_space
from math import cos, sin
from matplotlib import pyplot as plt

import numpy as np

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

plot_2d_phase_space(phase_space, "a periodic function")