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

        # Add a NOP operation if there are no available data to operate on
        if sink_i == len(sink_nodes):
            return ["nop", None, None, None], completed

        target_column = conn_mat[:, sink_i]
        source_is = target_column.nonzero()

        # Skip the sink node if it is an output node.
        if type(sink_nodes[sink_i]) != str:
            continue
        
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

def find_terminal_instrs(conn_mat, source_nodes, sink_nodes):
    # Finds the last instruction before a write to an output node.
    # The order that they are returned is the order of the output register.

    terminal_nodes = []
    for j, node in enumerate(sink_nodes):
        if type(node) != str:
            source_i = np.where(conn_mat[:, j] == 1.0)[0][0]
            terminal_nodes.append(source_nodes[source_i])
    
    return terminal_nodes

def update_reg_map(reg_map, core_instr, terminal_instrs, out_reg_offset, dbs):

    for i, reg in enumerate(reg_map):
        if reg['s'] == "avail" and core_instr[0] != 'nop':

            if core_instr[3] in terminal_instrs:
                reg_i = out_reg_offset + terminal_instrs.index(core_instr[3])
            else:
                reg_i = i
            
            instr_name = core_instr[0]
            reg_map[reg_i]["d"] = core_instr[3] # Set register data to result name.
            reg_map[reg_i]["dly"] = dbs["opcodes"][instr_name]["delay"] + 1 # Add one for the final register write
            reg_map[reg_i]["s"] = "wait"

            return reg_map

    return reg_map

def update_clk_cycle(reg_map, terminal_outs, completed_outs):

    for i, reg in enumerate(reg_map):
        if reg["s"] == 'wait':
            reg_map[i]['dly'] -= 1

            if reg_map[i]['dly'] == 0:
                if reg_map[i]['d'] in terminal_outs:
                    completed_outs += 1
                
                reg_map[i]['s'] = 'valid'
                del reg_map[i]['dly']
 
    return reg_map, completed_outs

def disp_reg_map(reg_map):

    for i, reg in enumerate(reg_map):
        print("{}: {}".format(i, reg))

def disp_exec_thread(instrs, index=0):

    for instr in instrs[index]:
        print("{}   {}, {}, {}".format(instr[0], instr[3], instr[1], instr[2]))
