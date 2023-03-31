from mimetypes import init
from re import sub
from tabnanny import verbose
import math
from sympy import Symbol, sympify, lambdify, symbols
import numpy as np
import scipy.integrate
from matplotlib import pyplot as plt
from parsers.equation_parse import substitute_subeqs, extract_equality_expr

def extract_vars(deriv_str):
    pre, _ = deriv_str.split('/')
    return pre[1:]

def parse_eqs(eq_str):
    # Parses input equation string.
    # TODO combine this with the start of the compiler pipeline.

    equations = extract_equality_expr(eq_str)
    system_eqs = substitute_subeqs(equations)

    # Vectorise input equations.
    input_symbols = []
    sys_dot = []
    deriv_strs = []
    sys_vars = []

    for lhs, rhs in system_eqs:

        sys_dot.append(rhs)
        deriv_strs.append(lhs)
        sys_vars.append(extract_vars(lhs))
        input_symbols += list(rhs.free_symbols)

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
    arg_vals = []
    for symbol in arg_symbols:
        try:
            arg_vals.append(arg_dict[str(symbol)])
        except:
            if str(symbol) == "e":
                arg_vals.append(math.e)
            else:
                raise Exception("System symbol {} not recognised".format(symbol))

    return arg_vals

def extract_var_class(nodes, filter_str):
    # Extract the output variables from a list of sink nodes.
    ret_nodes = []
    for node in nodes:
        if filter_str in str(node):
            ret_nodes.append(node)

    return ret_nodes

def run_pipeline(unit, in_state, args, dt, verbose=False):

    args['dt'] = dt

    # Prepare the initial state vector.
    source_state = np.zeros((len(unit.source_nodes)))
    source_valid = np.zeros((len(unit.source_nodes)))

    for i, node in enumerate(unit.source_nodes):
        if '_pre' in str(node):
            source_state[i] = in_state[str(node)[:-4]]
            source_valid[i] = 1
        elif '_const' in str(node):
            source_state[i] = float(str(node)[:-6])
            source_valid[i] = 1
        elif type(node) != str:
            source_state[i] = args[str(node)]
            source_valid[i] = 1
        else:
            source_state[i] = 0.0

    # Setup emulator state.
    val_mat    = np.zeros(np.shape(unit.conn_mat)) # The actual values in the emulated pipeline
    valid_mat  = np.zeros(np.shape(unit.conn_mat)) # Whether the value in that cell is acutually part of a valid computation.
    reg_mat    = np.zeros(np.shape(unit.conn_mat)) # The remaining register stages for these data

    val_vec     = np.zeros(np.shape(unit.sink_nodes)) # The values as they are delayed through the operator cells
    valid_vec   = np.zeros(np.shape(unit.sink_nodes)) # The values of the delays as the data moves through the pipeline.
    reg_vec     = np.zeros(np.shape(unit.sink_nodes)) # The remaining register stages for these data

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    terminate = False
    stage = 0
    while not terminate:
        if verbose:
            print("STAGE {}:".format(stage))
        stage += 1

        for j in range(len(valid_vec)):

            if valid_vec[j]:
                if reg_vec[j] == 0:
                    # Move data from operator array to input array.
                    node_name = unit.sink_nodes[j]
                    
                    node_i = list(unit.source_nodes).index(node_name)
                    source_state[node_i] = val_vec[j]
                    source_valid[node_i] = 1
                    if verbose:
                        print("{} moving from {} into matrix".format(val_vec[j], str(node_name)))
                    # Remove valid data from input array
                    val_vec[j] = 0.0
                    valid_vec[j] = 0.0
                else:
                    reg_vec[j] -= 1

        # Move the data into the pipeline.
        for i, (val, valid) in enumerate(zip(source_state, source_valid)):
            if valid:
                addrs = np.where(unit.conn_mat[i, :] > 0)[0]

                for addr in addrs:
                    val_mat[i, addr] = float(val)
                    valid_mat[i, addr] = 1
                    reg_mat[i, addr] = unit.predelays[i] + unit.dly_mat[i, addr]

                # Remove the data from input array
                source_state[i] = 0.0
                source_valid[i] = 0

        # Move relevant data from pipeline matrix into the operator array 
        # and decrement pipeline delays
        for j in range(len(unit.sink_nodes)):
            # Find addresses where data is valid.
            addrs = np.where(valid_mat[:, j])[0]

            in_args_vec = np.zeros((2))
            compute_result = False

            for addr in addrs:
                if reg_mat[:, j][addr] == 0:
                    # The delay has expired. Move data from routing matrix into input operator buffer
                    # # in order of precedence.
                    in_args_vec[int(unit.conn_mat[:, j][addr])-1] = val_mat[:, j][addr]
                    compute_result = True

                    # Delete the entry from the routing matrix.
                    val_mat[:, j][addr] = 0.0
                    valid_mat[:, j][addr] = 0.0
                elif reg_mat[:, j][addr] > 0:
                    reg_mat[:, j][addr] -= 1
            
            node_str = str(unit.sink_nodes[j]).split('_')[0]
            valid_req = True
            output_req = False
            if node_str in unit.arch_dbs.keys():
                if unit.arch_dbs[node_str]['input_num'] != len(addrs):
                    valid_req = False
            else:
                
                if len(addrs) == 1 and '_post' in str(unit.sink_nodes[j]):
                    output_req = True     

            if compute_result:
                
                if not valid_req:
                    pass
                else:
                    # Extract the computation in question.
                    if output_req:
                        node_str = str(unit.sink_nodes[j])
                        op_lambda = lambda a, b: a + b
                        target_delay = 0
                    else:
                        node_str = str(unit.sink_nodes[j]).split('_')[0]
                        target_op = unit.arch_dbs[node_str]['op']
                        target_delay = unit.arch_dbs[node_str]['delay']
                        input_num = unit.arch_dbs[node_str]['input_num']

                        if input_num == 1:
                            in_args_vec = in_args_vec[:1]

                        # TODO add operation parsing to remove the eval expression below
                        if unit.sink_nodes[j] in unit.assoc_dat.keys():
                            # The current node's operation is a lookup table emulation.
                            op_lambda = unit.assoc_dat[unit.sink_nodes[j]]
                        elif input_num == 1 and target_op[:9] != "lambda a:" or len(target_op) > 20:
                            raise Exception("MONARCH - Potentially unsafe operation detected")
                        elif input_num == 2 and target_op[:12] != "lambda a, b:" or len(target_op) > 20:
                            raise Exception("MONARCH - Potentially unsafe operation detected")
                        else:
                            op_lambda = eval(target_op)
                    
                    result = op_lambda(*in_args_vec)

                    if verbose:
                        print("{} produced {}, being delayed".format(str(unit.sink_nodes[j]), result))
                
                    # Move the result into the operator registers.
                    val_vec[j] = result
                    valid_vec[j] = 1
                    reg_vec[j] = target_delay - 1

                    # An output to an output node has been detected.
                    if type(unit.sink_nodes[j]) != str:
                        terminate = True

        if verbose:
            input()
            print("")
        
    # Extract the output states.
    outputs = [[str(x), y] for y, x in enumerate(unit.sink_nodes) if '_post' in str(x)]
    
    # TODO Improve how state is passed to and from this function.
    ret_dat = {}
    for out_lst in outputs:
        ret_dat[out_lst[0][:-5]] = val_vec[out_lst[1]]
    
    return ret_dat

def emulate_graph_unit(unit, time, dt, init_state, args):

    timesteps = int(float(time) / dt)

    output_vars = extract_var_class(unit.sink_nodes, '_post')

    ret_data = np.zeros([timesteps, len(output_vars)])
    state_var = init_state
    ret_data[0] = np.array(list(state_var.values()))

    for i in range(timesteps-1): 
        if (i+1) % 500 == 0:
            print("Time simulated: {:.2f} s".format(i * dt))  

        state_var = run_pipeline(unit, state_var, args, dt, verbose=False)
        ret_data[i+1] = np.array(list(state_var.values()))

    return ret_data

def pipeline_eumulator(eqs, graph_unit, initial_state, args, sim_time=10, dt=0.001):
    # Emulates a graph unit with register delays
    # with it's associated system of equations, and
    # reports output signal concordance.

    # Parse sympy equations.
    sys_dot, inputs, dim_names = parse_eqs(eqs)

    input_args = extract_arg_vals(args, inputs[2:])
    input_init_state = extract_arg_vals(initial_state, inputs[1])
    t_eval = np.linspace(0, sim_time, int(float(sim_time) / dt))

    # Numerically solve the system. TODO This solution is slightly inaccurate, find out why this is 
    sys_f = lambdify(inputs, sys_dot)
    solution = scipy.integrate.solve_ivp(sys_f, (0, sim_time), input_init_state, t_eval=t_eval, args=input_args)
    sim_dat = solution.y.T

    # Simulate the graph unit.
    pipe_dat = emulate_graph_unit(graph_unit, sim_time, dt, initial_state, args)

    # Compute the MSE between the two simulations.
    sub_dat = np.subtract(pipe_dat, sim_dat)
    square_dat = sub_dat ** 2
    mean_vec = np.mean(square_dat, axis=0)
    print("MONARCH - MSE for each system dimension: {}".format(mean_vec))

    plt.subplot(121)
    plt.title("Simulated Hardware Data")
    plt.plot(pipe_dat)

    plt.subplot(122)
    plt.title("Numerical Simulation Data")
    plt.plot(sim_dat)

    plt.show()

    """
    ax = plt.axes(projection ='3d')
    ax.plot3D(*np.transpose(sim_dat), 'red') 
    plt.show()
    """

def simulate_system(eqs, sys_data, sim_time=10, dt=0.001):

    # Parse sympy equations.
    sys_dot, inputs, dim_names = parse_eqs(eqs)

    input_args = extract_arg_vals(sys_data, inputs[2:])
    input_init_state = extract_arg_vals(sys_data, inputs[1])
    t_eval = np.linspace(0, sim_time, int(float(sim_time) / dt))

    # Numerically solve the system. 
    sys_f = lambdify(inputs, sys_dot)
    solution = scipy.integrate.solve_ivp(sys_f, (0, sim_time), input_init_state, t_eval=t_eval, args=input_args)
    sim_dat = solution.y.T

    print(sim_dat[:10])
    plt.title("Numerical Simulation Data")
    plt.plot(sim_dat)

    plt.show()