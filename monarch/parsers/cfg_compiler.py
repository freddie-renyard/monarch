import json
import os
import numpy as np

def get_symbols(cfg):
    
    ret_lst = []
    for input_node in cfg['inputs']:
        if type(input_node) != dict:
            # A valid symbol has been found.
            ret_lst += [input_node]
        else:
            ret_lst += get_symbols(input_node)

    return ret_lst

def verify_graph(cfg, dbs):

    if type(cfg) != dict:
        return True
    
    if cfg['op'] not in dbs.keys():
        return False
    else:
        valid_graph = True
        for input_node in cfg['inputs']:
            valid_graph = verify_graph(input_node, dbs)
            if not valid_graph:
                return False
            
        return valid_graph

def get_longest_path(cfg, dbs, depth=0):

    if type(cfg) != dict:
        return 0

    # Find the longest branch which is associated with
    # the current node.
    max_len = 0
    for input_node in cfg['inputs']:
        path_len = get_longest_path(input_node, dbs, depth)
        if path_len > max_len:
            max_len = path_len
    
   # Add the length of the longest branch to the current node's delay. 
    return max_len + dbs[cfg['op']]['delay']

def build_conn_mat(cfg, source_nodes, sink_nodes, conn_mats=None):

    # Construct the connectivity matrices on entry function call.
    if conn_mats is None:
        conn_mats = {
            "conn": np.zeros([len(source_nodes), len(sink_nodes)]),
            "delay": np.zeros([len(source_nodes), len(sink_nodes)])
        }

    # Construct the matrices from the top (root) node of the graph down.
    sink_i = sink_nodes.index(cfg['op']) # TODO add duplicate checking to input lists.
    source_is = []
    for input_node in cfg['inputs']:
        if type(input_node) == dict:
            # Subnode detected. Extract opcode as output.
            node_i = source_nodes.index(input_node['op'])

            # Recursively repeat to build the connectivity matrix.
            conn_mats = build_conn_mat(input_node, source_nodes, sink_nodes, conn_mats=conn_mats)
        else:
            node_i = source_nodes.index(input_node)
        source_is.append(node_i)
    
    # Add the computed graph indices into the connectivity matrix
    for i, source_i in enumerate(source_is):
        conn_mats['conn'][source_i, sink_i] = i + 1

    return conn_mats

def modify_ops(cfg, id=0):

    if type(cfg) != dict:
        return cfg

    cfg['op'] += "_{}".format(id)

    for i, input_node in enumerate(cfg['inputs']):
        if type(input_node) == dict:
            id += 1
            cfg['inputs'][i] = modify_ops(input_node, id=id)

    return cfg

def extract_source_nodes(cfg):

    if type(cfg) != dict:
        # Source node found.
        return [cfg]

    source_nodes = []
    for input_node in cfg['inputs']:
        source_nodes += extract_source_nodes(input_node)
    
    return source_nodes

def extract_sink_nodes(cfg):

    if type(cfg) != dict:
        # Source node found.
        return []

    sink_nodes = [cfg['op']]
    for input_node in cfg['inputs']:
        if type(input_node) == dict:
            sink_nodes += extract_sink_nodes(input_node)
    
    return sink_nodes

def cfg_to_mats(cfg, dbs):

    # Get all unique symbols in graph.
    symbols = get_symbols(cfg)
    symbols = list(set(symbols))

    # Append a unique identifier to every operator node in graph.
    cfg = modify_ops(cfg)

    # Get all the unique identifiers in the graph.
    source_nodes = extract_source_nodes(cfg)
    inout_nodes = extract_sink_nodes(cfg)

    source_nodes = [*source_nodes, *inout_nodes]
    sink_nodes = [*inout_nodes, 'root']

    # Build flow up from the deepest node in the graph,
    # where both nodal inputs are symbols.
    print(cfg)
    print(source_nodes, sink_nodes)
    conn_mats = build_conn_mat(cfg, source_nodes, sink_nodes)

    # Add the output root node onto the connectivity matrices
    root_i = sink_nodes.index('root')
    source_i = source_nodes.index(cfg['op']) # Final CFG operation
    conn_mats['conn'][source_i, root_i] = 1

    print("CONN + PRECEDENCE:")
    print(conn_mats['conn'], end="\n\n")
    print("DELAY:")
    print(conn_mats['delay'], end="\n\n")
    
    exit()

def cfg_to_pipeline(eq_system):

    # Extract all graph input symbols (variables, constant-variables, and constants).
    symbols = []
    for eq in eq_system:
        symbols += get_symbols(eq["cfg"])

    input_set = set(symbols)

    script_dir = os.path.dirname(__file__)
    rel_path = "arch_dbs.json"
    abs_file_path = os.path.join(script_dir, rel_path)

    # Open architecture database.
    with open(abs_file_path) as file:
        dbs = json.loads(file.read())
        dbs = dbs['opcodes']
    
    # Verify that the graph contains only legal operations.
    for eq in eq_system:
        valid = verify_graph(eq["cfg"], dbs)
        if not valid:
            Exception("MONARCH - Unsupported operations present in compiled graphs")

    """
    # Stage 1: Find longest path in the total system graph.
    longest_path = 0
    for eq in eq_system:
        path_len = get_longest_path(eq["cfg"], dbs)
        print(path_len)
    """

    # Stage 1: compile each individual graph to it's matrix representation.
    for eq in eq_system:
        conn_mat = cfg_to_mats(eq["cfg"], dbs) 
        exit()