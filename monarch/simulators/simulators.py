from sympy import Symbol, sympify, lambdify, symbols
import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt

def extract_vars(deriv_str):

    pre, _ = deriv_str.split('/')
    return pre[1:]

def parse_eqs(eq_str):
    # Parses input equation string.
    # TODO combine this with the start of the compiler pipeline.

    substrs = eq_str.split('\n')
    substrs = [x.strip() for x in substrs if x.strip() != '']
    
    # Vectorise input equations.
    input_symbols = []
    sys_dot = []
    deriv_strs = []
    sys_vars = []
    for substr in substrs:
        deriv_str, subeq_str = substr.split(' = ')

        subeq = sympify(subeq_str)
        sys_dot.append(subeq)
        deriv_strs.append(deriv_str)
        sys_vars.append(extract_vars(deriv_str))
        input_symbols += list(subeq.free_symbols)

    input_symbols = list(set(input_symbols))

    # Vectorise the input variables.
    sys_vec = []
    for in_var in sys_vars:
        symbol = Symbol(in_var)
        sys_vec.append(symbol)
        if symbol in input_symbols:
            input_symbols.remove(symbol)

    sys_vec = tuple(sys_vec)

    # Add time variable
    t = symbols('t')
    ret_vars = (t, sys_vec, *input_symbols)

    return sys_dot, ret_vars, deriv_strs

def extract_arg_vals(arg_dict, arg_symbols):
    return [arg_dict[str(symbol)] for symbol in arg_symbols]

def pipeline_eumulator(eqs, graph_unit, initial_state, args, sim_time=10, dt=0.001):
    # Emulates a graph unit with register delays
    # with it's associated system of equations, and
    # reports output signal concordance.

    # Parse sympy equations.
    sys_dot, inputs, dim_names = parse_eqs(eqs)

    input_args = extract_arg_vals(args, inputs[2:])
    t_eval = np.linspace(0, sim_time, int(float(sim_time) / dt))

    # Numerically solve the system.
    sys_f = lambdify(inputs, sys_dot)
    solution = scipy.integrate.solve_ivp(sys_f, (0, sim_time), initial_state, t_eval=t_eval, args=input_args)

    ax = plt.axes(projection ='3d')
    ax.plot3D(*np.transpose(solution.y.T), 'green') 
    plt.show()
