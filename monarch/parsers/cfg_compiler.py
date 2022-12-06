import json
import os
from parsers.hardware_objs import GraphUnit
import numpy as np
from matplotlib import pyplot as plt
from sympy import Symbol

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

def build_conn_mat(cfg, source_nodes, sink_nodes, conn_mat=None):

    # Construct the connectivity matrices on entry function call.
    if conn_mat is None:
        conn_mat = np.zeros([len(source_nodes), len(sink_nodes)])
    # Construct the matrices from the top (root) node of the graph down.
    sink_i = sink_nodes.index(cfg['op']) # TODO add duplicate checking to input lists.
    source_is = []
    for input_node in cfg['inputs']:
        if type(input_node) == dict:
            # Subnode detected. Extract opcode as output.
            node_i = source_nodes.index(input_node['op'])

            # Recursively repeat to build the connectivity matrix.
            conn_mat = build_conn_mat(input_node, source_nodes, sink_nodes, conn_mat=conn_mat)
        else:
            node_i = source_nodes.index(input_node)
        source_is.append(node_i)
    
    # Add the computed graph indices into the connectivity matrix
    for i, source_i in enumerate(source_is):
        conn_mat[source_i, sink_i] = i + 1

    return conn_mat

def build_delay_mat(cfg, source_nodes, sink_nodes, arch_dbs, dly_mat=None):
    # Construct the pipeline delay matrix. 

    # Construct the matrix on entry function call.
    if dly_mat is None:
        dly_mat = np.zeros([len(source_nodes), len(sink_nodes)])

    opcode = cfg['op'].split("_")[0]
    inherent_dly = arch_dbs[opcode]['delay']

    branch_dlys = []
    sink_i = sink_nodes.index(cfg['op']) # TODO add duplicate checking to input lists.
    source_is = []
    for input_node in cfg['inputs']:
        if type(input_node) == dict:
            # The node is a subtree, so the delay must be computed.
            node_i = source_nodes.index(input_node['op'])
            dly_mat, branch_delay = build_delay_mat(
                input_node, 
                source_nodes, 
                sink_nodes,
                arch_dbs,
                dly_mat=dly_mat
            )
        else:  
            # The node is an input, so nodal delay is 0.  
            node_i = source_nodes.index(input_node)         
            branch_delay = 0
        source_is.append(node_i)
        branch_dlys.append(branch_delay)

    unbalanced = branch_dlys.count(branch_dlys[0]) != len(branch_dlys)
    if unbalanced:
        # Get the index of the unbalanced path. TODO only works for binary trees.
        low_i = branch_dlys.index(min(branch_dlys))
        dly_mat[source_is[low_i], sink_i] = max(branch_dlys) - min(branch_dlys)
        total_dly = inherent_dly + max(branch_dlys)
    else:
        total_dly = inherent_dly + branch_dlys[0]
    
    return dly_mat, total_dly

def modify_ops(cfg, id=0):

    if type(cfg) != dict:
        return cfg

    cfg['op'] += "_{}".format(id)

    max_id = 0
    for i, input_node in enumerate(cfg['inputs']):
        if type(input_node) == dict:
            id += 1
            cfg['inputs'][i], max_id = modify_ops(input_node, id=id)

    if id > max_id:
        max_id = id
    return cfg, max_id

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

def cfg_to_mats(cfg, output_var, dbs, start_id):

    # Get all unique symbols in graph.
    symbols = get_symbols(cfg)
    symbols = list(set(symbols))

    # Append a unique identifier to every operator node in graph.
    cfg, max_id = modify_ops(cfg, id=start_id)

    # Get all the unique identifiers in the graph.
    source_nodes = extract_source_nodes(cfg)
    inout_nodes = extract_sink_nodes(cfg)

    source_nodes = [*source_nodes, *inout_nodes]
    sink_nodes = [*inout_nodes, output_var]
    print(source_nodes)
    for i, node in enumerate(source_nodes):
        if type(node) != str:
            if node.is_number:
                source_nodes[i] = Symbol(str(node) + "_const")

    # Build flow up from the deepest node in the graph,
    # where both nodal inputs are symbols.
    conn_mat = build_conn_mat(cfg, source_nodes, sink_nodes)
    dly_mat, max_dly = build_delay_mat(cfg, source_nodes, sink_nodes, dbs)
    
    # Add the output root node onto the connectivity matrices
    root_i = sink_nodes.index(output_var)
    source_i = source_nodes.index(cfg['op']) # Final CFG operation
    conn_mat[source_i, root_i] = 1
    
    return [conn_mat, dly_mat, source_nodes, sink_nodes, max_dly], max_id + 1

def paste_matrices(mat_1, mat_2):

    mat_1_shape = np.shape(mat_1)
    mat_2_shape = np.shape(mat_2)

    new_mat = np.zeros(np.add(mat_1_shape, mat_2_shape))

    # Paste first matrix
    new_mat[0:mat_1_shape[0], 0:mat_1_shape[1]] = mat_1
    new_mat[mat_1_shape[0]:, mat_1_shape[1]:] = mat_2
    
    return new_mat

def combine_trees(system_data):
    
    conn_mat, dly_mat, source_nodes, tmp, max_dly = system_data[0]
    sink_nodes = tmp.copy() # Prevent linking the first equation to the concatenations below

    for dat in system_data[1:]:
        sub_conn, sub_dly, sub_source_nodes, sub_sink_nodes, sub_max_dly = dat

        if sub_max_dly > max_dly:
            max_dly = sub_max_dly
        
        # Stage 1: combine all matrices and nodes into 1 matrix.
        # TODO add output variable naming much eariler on in pipe
        source_nodes += sub_source_nodes
        sink_nodes += sub_sink_nodes
        conn_mat = paste_matrices(conn_mat, sub_conn)
        dly_mat = paste_matrices(dly_mat, sub_dly)

    # Add output delay registers to equalise pipeline depth for each tree.
    for dat in system_data:
        _, _, _, sink_nodes_temp, tree_dly = dat
        val = [x for x in sink_nodes_temp if "_post" in str(x)][0]
        diff_from_max = max_dly - tree_dly
        
        for i, node in enumerate(sink_nodes):
            if str(node) == str(val):
                break
        
        input_node_i = list(conn_mat[:, i] > 0).index(True)
        dly_mat[input_node_i, i] = diff_from_max
        
    return GraphUnit(
        conn_mat,
        dly_mat,
        source_nodes,
        sink_nodes
    )

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

    # Stage 1: compile each individual graph to it's matrix representation.
    start_id = 0
    ret_vals = []
    for eq in eq_system:
        ret_data, start_id = cfg_to_mats(eq["cfg"], eq['root'], dbs, start_id)
        ret_vals.append(ret_data)    

    # Stage 2: Combine the matrices and associated nodes for the full system.
    compiled_unit = combine_trees(ret_vals)

    # Stage 3: Run cleanup/optimisation algorithms on the graph unit
    compiled_unit.arch_dbs = dbs # Add the architecture spec to the unit

    compiled_unit.combine_vars()
    compiled_unit.reorder_source_nodes()
    compiled_unit.reorder_sink_nodes()

    # TODO Compute the max pipeline depth of the total graph.

    compiled_unit.compute_predelay()

    return compiled_unit