import numpy as np

def allocate_core_instr(conn_mat, source_nodes, sink_nodes, completed=[]):
    # Searches through the source nodes and allocates an instruction
    # based on available operands.
    
    comp_source_ops = True
    sink_i = -1
    while comp_source_ops:
        sink_i += 1
        target_column = conn_mat[:, sink_i]
        source_is = target_column.nonzero()
        for ind in source_is[0]:
            if type(source_nodes[ind]) != str:
                if sink_i not in completed:
                    comp_source_ops = False
            else:
                comp_source_ops = True
    
    op = sink_nodes[sink_i].split("_")[0]
    in_0 = source_nodes[np.where(target_column == 1.0)][0]
    in_1 = source_nodes[np.where(target_column == 2.0)][0]
    completed.append(sink_i)

    return [op, in_0, in_1], completed

