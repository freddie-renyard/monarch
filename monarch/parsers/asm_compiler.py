import numpy as np

def allocate_core_instr(conn_mat, source_nodes, sink_nodes, reg_map, completed=[]):
    # Searches through the source nodes and allocates an instruction
    # based on available operands.

    # Get the list of available results that can be computed.
    # If lock, the node is an input. If valid, it is an intermediate result that has been computed.
    avail_dat = [x["d"] for x in reg_map if x["d"] if x["s"] == 'lock' or x["s"] == "valid"]

    comp_source_ops = True
    sink_i = -1
    while comp_source_ops:
        sink_i += 1
        target_column = conn_mat[:, sink_i]
        source_is = target_column.nonzero()
        
        # Check each of the input source nodes for the column to
        # see if their results are available.
        for ind in source_is[0]:
            if str(source_nodes[ind]) in avail_dat:
                if sink_i not in completed:
                    comp_source_ops = False
            else:
                comp_source_ops = True
    
    op   = sink_nodes[sink_i].split("_")[0]
    in_0 = source_nodes[np.where(target_column == 1.0)][0]
    in_1 = source_nodes[np.where(target_column == 2.0)][0]
    out  = sink_nodes[sink_i]
    completed.append(sink_i)

    return [op, in_0, in_1, out], completed

