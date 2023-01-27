from distutils import core
from pickletools import opcodes
from shutil import ExecError
import numpy as np
from parsers.bin_compiler import convert_to_uint

def allocate_core_instr(conn_mat, source_nodes, sink_nodes, reg_map, assoc_dat, avail_consts, dbs, primaries=[], completed=[]):
    # Searches through the source nodes and allocates an instruction
    # based on available operands.

    # Get the list of available results that can be computed.
    # If valid, it is an intermediate result that has been computed.
    avail_dat = [x["d"] for x in reg_map if x["s"] == "valid"]
    avail_dat += avail_consts

    # Determine the order of evaluation for faster execution 
    sink_is = []
    for prim_node in primaries:
        sink_is.append(
            np.where(sink_nodes == prim_node)[0][0]
        )
    sink_is += [x for x in range(len(sink_nodes)) if x not in sink_is]

    comp_source_ops = True
    i = -1
    while comp_source_ops:
        i += 1

        # Add a NOP operation if there are no available data to operate on
        if i >= len(sink_nodes):
            return ["nop", None, None, None], completed

        sink_i = sink_is[i]

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
                break       

    op = sink_nodes[sink_i].split("_")[0]

    if dbs['opcodes'][op]['input_num'] == 1:
        if op == 'square':
            in_0 = source_nodes[target_column.nonzero()][0]
            in_1 = in_0
        elif op == 'lut':
            if str(sink_nodes[sink_i]) in assoc_dat.keys():
                in_0 = source_nodes[target_column.nonzero()][0]
                try:
                    in_1 = dbs["lut_functions"][assoc_dat[sink_nodes[sink_i]]]['subopcode']
                except:
                    raise Exception("Exponent base {} not currently supported for lut instruction.".format(assoc_dat[sink_nodes[sink_i]]))
            else:
                raise Exception("Associated LUT data not found for {}".format(sink_nodes[sink_i]))
        else:
            raise Exception("MONARCH - This 1-input instruction isn't supported: {}".format(op))
    else:
        in_0 = source_nodes[np.where(target_column == 1.0)][0]
        in_1 = source_nodes[np.where(target_column == 2.0)][0]

    out  = sink_nodes[sink_i]
    completed.append(sink_i)

    return [op, in_0, in_1, out], completed

def find_terminal_instrs(conn_mat, source_nodes, sink_nodes):
    # Finds the last instruction before a write to an output node.
    # The order that they are returned is the order of the output register.

    terminal_nodes = []
    assoc_names = []
    for j, node in enumerate(sink_nodes):
        if type(node) != str:
            source_i = np.where(conn_mat[:, j] == 1.0)[0][0]
            terminal_nodes.append(source_nodes[source_i])
            assoc_names.append(sink_nodes[j])
    
    return terminal_nodes, assoc_names

def update_reg_map(reg_map, core_instr, terminal_instrs, terminal_vars, out_reg_offset, dbs, out_regs={}):

    for i, reg in enumerate(reg_map):
        if reg['s'] == "avail" and core_instr[0] != 'nop':

            if core_instr[3] in terminal_instrs:
                reg_i = out_reg_offset + terminal_instrs.index(core_instr[3])
                out_regs[str(terminal_vars[terminal_instrs.index(core_instr[3])])] = reg_i
            else:
                reg_i = i
            
            instr_name = core_instr[0]
            reg_map[reg_i]["d"] = core_instr[3] # Set register data to result name.
            reg_map[reg_i]["dly"] = dbs["opcodes"][instr_name]["delay"] + 1 # Add one for the final register write
            reg_map[reg_i]["s"] = "wait"

            return reg_map, out_regs

    return reg_map, out_regs

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
        print("{}\t {}, {}, {}".format(instr[0], instr[3], instr[1], instr[2]))

def instr_to_asm(instr, reg_map, const_names):

    instr = [str(x) for x in instr] # Ensure instruction is stringified.

    op = instr[0]
    regs = [None, None, None]

    # Ensure that the subopcode for the lut instruction is passed through.
    if instr[0] == 'lut':
        regs[1] = int(instr[2])

    for i, input_operand in enumerate(instr[1:3]):
        if input_operand in const_names:
            op_is = np.where(np.array(instr[1:3]) == input_operand)[0]
            for op_i in op_is:
                regs[op_i] = "c{}".format(const_names.index(input_operand))

    for i, reg in enumerate(reg_map):
        if reg["d"] in instr[1:]:
            op_is = np.where(np.array(instr[1:]) == reg["d"])[0]
            for op_i in op_is:
                regs[op_i] = "r{}".format(i)

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
                    ['nop', None, nop_ctr, None]
                )
                new_asm.append(
                    instr
                )
                nop_ctr = 0
            else:
                new_asm.append(instr)
    if nop_ctr:
        new_asm.append(
            ['nop', None, nop_ctr, None]
        )

    return new_asm

def preprocess_asm(core_asm):

    for i, instr in enumerate(core_asm):
        if instr[0] != 'nop':
            for op_i, operand in enumerate(instr[1:3]):
                if 'c' in str(operand):
                    if op_i == 0:
                        # The constant is in the LSBs of the instruction,
                        # as it has operator precedence
                        core_asm[i][0] += '_cl'
                    elif op_i == 1:
                        core_asm[i][0] += '_cm'

    return core_asm

def ptr_to_bin(reg, width):

    if "r" in reg:
        # A register reference is being compiled.
        return convert_to_uint(int(reg.split("r")[1]), width)
    elif "c" in reg:
        # A reference to a constant is being compiled.
        return convert_to_uint(int(reg.split("c")[1]), width)

    raise Exception("MONARCH - Unsupported reference {}".format(reg))

def instr_to_machcode(instr, dbs):

    machcode = ''

    op = instr[0]
    try:
        op_dat = dbs["isa"][op]
    except:
        raise Exception("MONARCH - Unsupported operation '{}'.".format(op))

    machcode += convert_to_uint(op_dat["opcode"], dbs["manycore_params"]["machcode_params"]['op_width'])

    reg_width = dbs["manycore_params"]["machcode_params"]['reg_ptr_width']
    if op_dat["type"] in ["3r", "2rc", "2cr"]:
        # This instruction has two input register operands and one output register.
        in_reg_0_bin = ptr_to_bin(instr[1], reg_width)
        in_reg_1_bin = ptr_to_bin(instr[2], reg_width)
        out_reg_bin  = ptr_to_bin(instr[3], reg_width)

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
    elif op_dat['type'] == '2rl':
        in_reg_0_bin = ptr_to_bin(instr[1], reg_width)
        subopcode = convert_to_uint(instr[2], reg_width)
        out_reg_bin  = ptr_to_bin(instr[3], reg_width)
        machcode = out_reg_bin + subopcode + in_reg_0_bin + machcode        

    return machcode + "\n"

def get_branches(target_node, conn_mat, source_nodes, sink_nodes):
    # Returns the non-terminal branches of a node

    node_i = np.where(sink_nodes == target_node)[0][0]
    target_is = conn_mat[:, node_i].nonzero()
    new_target_nodes = source_nodes[target_is]

    temp = [target_node]
    for new_target in new_target_nodes:
        if type(new_target) == str:
            temp += get_branches(
                new_target, conn_mat, source_nodes, sink_nodes
            )

    return temp

def determine_primary(conn_mat, source_nodes, sink_nodes, ignore_lst=[], target_op="div"):
    # Determines a list of intermediates that should be completed
    # before any others. This is to allow for better use of ALU time:
    # completing work for time-consuming units like the division block 
    # first, so that lower-latency work can be done whilst this result is computed.

    # Determine the intermediate that will be targeted.
    target_node = None
    for node in source_nodes:
        if type(node) == str:
            if target_op + '_' in node:
                if node not in ignore_lst:
                    target_node = node
                    break
    
    if target_node is not None:
        return get_branches(
            target_node,
            conn_mat,
            source_nodes,
            sink_nodes
        )[::-1]
    else:
        return []