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

    # Stage 1: Find longest path in the total system graph.
    longest_path = 0
    for eq in eq_system:
        path_len = get_longest_path(eq["cfg"], dbs)
        print(path_len)
        exit()