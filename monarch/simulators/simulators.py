from mimetypes import init
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

def run_pipeline(unit, in_state, args, dt):

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
    while not terminate:
        
        # Move the data into the pipeline.
        for i, (val, valid) in enumerate(zip(source_state, source_valid)):
            if valid:
                addrs = np.where(unit.conn_mat[i, :] > 0)[0]
                for addr in addrs:
                    val_mat[i, addr] = float(val)
                    valid_mat[i, addr] = 1
                    reg_mat[i, addr] = unit.predelays[i] + unit.dly_mat[i, addr]
        
        print(val_mat, reg_mat)

        for j in range(len(unit.sink_nodes)):
            # Find addresses where data is valid.
            addrs = np.where(valid_mat[:, j] > 0)[0]
            
            in_args_vec = np.zeros((2))
            for addr in addrs:
                if reg_mat[j, addr] == 0:
                    # The delay has expired. Move data from routing matrix into input operator buffer
                    # # in order of precedence.
                    
                    in_args_vec[int(unit.conn_mat[j, addr])-1] = val_mat[j, addr]

                    # Delete the entry from the routing matrix.
                    val_mat[j, addr] = 0.0
                    valid_mat[j, addr] = 0.0
                elif reg_mat[j, addr] > 0:
                    reg_mat[j, addr] -= 1
            
            print(in_args_vec)
    
        unit.show_report()

        exit()
        

def emulate_graph_unit(unit, time, dt, init_state, args):

    timesteps = int(float(time) / dt)

    output_vars = extract_var_class(unit.sink_nodes, '_post')

    ret_data = np.zeros((len(output_vars), timesteps))
    state_var = init_state

    for i in range(timesteps):   
        
        state_var = run_pipeline(unit, state_var, args, dt)
        pass

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

    # Simulate the grpah unit.
    emulate_graph_unit(graph_unit, sim_time, dt, initial_state, args)





    exit()
    ax = plt.axes(projection ='3d')
    ax.plot3D(*np.transpose(solution.y.T), 'green') 
    plt.show()
