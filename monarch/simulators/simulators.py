from mimetypes import init
from tabnanny import verbose
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

def extract_var_class(nodes, filter_str):
    # Extract the output variables from a list of sink nodes.
    ret_nodes = []
    for node in nodes:
        if filter_str in str(node):
            ret_nodes.append(node)

    return ret_nodes

def run_pipeline(unit, in_state, args, dt, verbose=False):

    args['dt'] = dt
    input_nodes = extract_var_class(unit.source_nodes, '_pre')

    # Prepare the initial state vector.
    source_state = np.zeros((len(unit.source_nodes)))
    source_valid = np.zeros((len(unit.source_nodes)))
    for i, node in enumerate(unit.source_nodes):
        if '_pre' in str(node):
            source_state[i] = in_state[str(node)[:-4]]
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

            # Catch the invalid result, where only one of two essential inputs to a binary node is valid
            if compute_result and len(addrs) == 1 and '_post' not in str(unit.sink_nodes[j]):
                pass
            else:
                if compute_result:
                    # Extract the computation in question.
                    if compute_result and len(addrs) == 1:
                        node_str = str(unit.sink_nodes[j])
                        target_op = lambda a: a
                        target_delay = 0
                    else:
                        node_str = str(unit.sink_nodes[j]).split('_')[0]
                        target_op = unit.arch_dbs[node_str]['op']
                        target_delay = unit.arch_dbs[node_str]['delay']

                        # TODO add operation parsing to remove the eval expression below
                        if target_op[:12] != "lambda a, b:" or len(target_op) > 20:
                            raise Exception("MONARCH - Potentially unsafe operation detected")
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

    for i in range(timesteps): 
        if (i+1) % 500 == 0:
            print("Time simulated: {:.2f} s".format(i * dt))  

        state_var = run_pipeline(unit, state_var, args, dt, verbose=False)
        ret_data[i] = np.array(list(state_var.values()))

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

    # Numerically solve the system.
    sys_f = lambdify(inputs, sys_dot)
    solution = scipy.integrate.solve_ivp(sys_f, (0, sim_time), input_init_state, t_eval=t_eval, args=input_args)
    sim_dat = solution.y.T

    # Simulate the graph unit.
    pipe_dat = emulate_graph_unit(graph_unit, sim_time, dt, initial_state, args)
    
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

