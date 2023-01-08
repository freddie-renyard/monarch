import numpy as np
from parsers.bin_compiler import convert_to_uint

def allocate_core_instr(conn_mat, source_nodes, sink_nodes, reg_map, completed=[]):
    # Searches through the source nodes and allocates an instruction
    # based on available operands.

    # Get the list of available results that can be computed.
    # If valid, it is an intermediate result that has been computed.
    avail_dat = [x["d"] for x in reg_map if x["d"] if x["s"] == "valid"]

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

def find_stale_results(reg_map, conn_mat, source_nodes, sink_nodes, completed):

    stringified_source = [str(x) for x in source_nodes]
    for i, reg in enumerate(reg_map):
        if reg["d"] in stringified_source:
            # Extract dependent operations
            source_i     = stringified_source.index(reg['d'])
            dependent_is = conn_mat[source_i, :].nonzero()[0]
            
            complete_is = [x for x in dependent_is if x in completed]
            
            if len(complete_is) == len(dependent_is):
                # All of the dependents have been computed. Dereference the register
                reg_map[i]['s'] = "avail"
                reg_map[i]['d'] = None

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

def instr_to_asm(instr, reg_map):

    instr = [str(x) for x in instr] # Ensure instruction is stringified.

    op = instr[0]
    regs = [None, None, None]
    
    for i, reg in enumerate(reg_map):
        if reg["d"] in instr[1:4]:
            op_index = instr[1:4].index(reg["d"])
            regs[op_index] = "r{}".format(i)

    return [op, *regs]

def collapse_nops(asm):

    nop_ctr = 0
    new_asm = []
    for instr in asm:
        if instr[0] == "nop":
            nop_ctr += 1
        else:
            if nop_ctr:
                new_asm.append(
                    ['nop', None, nop_ctr-1, None]
                )
                new_asm.append(
                    instr
                )
                nop_ctr = 0
            else:
                new_asm.append(instr)

    if nop_ctr:
        new_asm.append(
            ['nop', None, nop_ctr-1, None]
        )

    return new_asm

def reg_ptr_to_bin(reg, width):
    return convert_to_uint(int(reg.split("r")[1]), width)

def instr_to_machcode(instr, dbs):

    machcode = ''

    op = instr[0]
    try:
        op_dat = dbs["isa"][op]
    except:
        raise Exception("MONARCH - Unsupported operation '{}'.".format(op))

    machcode += convert_to_uint(op_dat["opcode"], dbs["manycore_params"]["machcode_params"]['instr_width'])

    reg_width = dbs["manycore_params"]["machcode_params"]['reg_ptr_width']
    if op_dat["type"] == "3r":
        # This instruction has two input register operands and one output register.
        
        in_reg_0_bin = reg_ptr_to_bin(instr[1], reg_width)
        in_reg_1_bin = reg_ptr_to_bin(instr[2], reg_width)
        out_reg_bin  = reg_ptr_to_bin(instr[3], reg_width)

        machcode = out_reg_bin + in_reg_1_bin + in_reg_0_bin + machcode
    elif op_dat['type'] == "0r":
        # The instruction takes no register operands.
        if op == 'nop':
            subopcode = convert_to_uint(op_dat['subopcode'], reg_width) 
            dly_time = convert_to_uint(instr[2], reg_width)
            machcode = '0'*reg_width + dly_time + subopcode + machcode
        elif op == "halt":
            subopcode = convert_to_uint(op_dat['subopcode'], reg_width) 
            machcode = '0'*reg_width*2 + subopcode + machcode

    return machcode + "\n"