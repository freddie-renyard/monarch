import json
import os

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

def build_conn_mat(cfg):

    deepest_op = True
    for input_node in cfg['inputs']:
        if type(input_node) == dict:
            deepest_op = False
            build_conn_mat(input_node)
    
    conn_mat = []
    source_nodes = []
    sink_nodes = []
    if deepest_op:
        source_nodes += cfg['inputs']
        sink_nodes += [cfg['op']]
        print(source_nodes, sink_nodes)
        


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
    sink_nodes = extract_sink_nodes(cfg) 

    # Build flow up from the deepest node in the graph,
    # where both nodal inputs are symbols.
    conn_mat = build_conn_mat(cfg)

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