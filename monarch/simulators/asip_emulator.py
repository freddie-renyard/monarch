import os
import json
from parsers.float_compiler import BinCompiler
import numpy as np 

def parse_machcode(file_path, isa):
    # Parse the input machine code file
        
    with open(file_path) as file:
        binary = file.read()
        machcodes = binary.split("\n")[1:-1]
        
        masm = []
        for machcode in machcodes:
            opcode = int(machcode[-5:], 2)
            source_1 = int(machcode[-10:-5], 2)
            source_2 = int(machcode[-15:-10], 2)
            dest_reg = int(machcode[0:-15], 2)
            
            candidates = []
            for op in isa.keys():
                if isa[op]["opcode"] == opcode:
                    candidates.append(op)

            # Disambiguate operation candidates
            if 'square' in candidates:
                op = 'mult'
            elif 'halt' in candidates:
                subopcode = source_1
                if subopcode == 1:
                    op = 'halt'
                else:
                    op = 'nop'
            elif len(candidates) == 1:
                op = candidates[0]
            else:
                raise Exception("MONARCH - Unsupported instruction in machcode")
            
            masm.append({
                "op": op,
                "s1": source_1,
                "s2": source_2,
                "dest": dest_reg
            })
        
    return masm

def parse_consts(file_path, n_man, n_exp):

    with open(file_path) as file:
        binary = file.read()
    
    consts_bin = binary.split('\n')[:-1]
    return [BinCompiler.decode_custom_float(x, n_man, n_exp) for x in consts_bin]
    
def parse_regs(regrefs, memfile, n_man, n_exp, reg_num):

    with open(regrefs) as file:
        binary = file.read()
        regs_bin = binary.split('\n')[:-1]
    
    with open(memfile) as file:
        binary = file.read()
        mem_bin = binary.split('\n')[:-1]

    regs = [int(x, 2) for x in regs_bin]
    data = [BinCompiler.decode_custom_float(x, n_man, n_exp) for x in mem_bin]
    reg_map = [None for x in range(reg_num)]

    for reg_i in regs:
        # This corresponds to the 'Read' step in hardware
        reg_map[reg_i] = data[0]
        data = data[1:]

    return reg_map

def execute_masm(masm, reg_table, const_table, arch_ops):

    clk = 0
    run = True
    pc = 0

    # Construct a table to check for if a result has been utilised
    utilised_table = [(x is None) for x in reg_table]

    print("START REGISTER MAP:")
    for i, dat in enumerate(reg_table):
        print("r{}  {}".format(i, dat))

    print("START CONSTANT MAP:")
    for i, dat in enumerate(const_table):
        print("c{}  {}".format(i, dat))

    dly_res = []
    dly_ctr = 0
    while run:

        # Stall execution if the nop counter is high
        if dly_ctr == 0:

            instr = masm[pc]
            
            # Decode the instruction
            opcode_split = instr["op"].split("_")

            # Decode sources
            opcode = opcode_split[0]
            if len(opcode_split) == 1:
                # Data source is register
                s1, s2 = reg_table[instr['s1']], reg_table[instr['s2']]
                utilised_table[instr['s1']] = True
                utilised_table[instr['s2']] = True
                active_regs = [instr['s1'], instr['s2']]
            else:
                if opcode_split[1] == 'cl':
                    s1, s2 = const_table[instr['s1']], reg_table[instr['s2']]
                    utilised_table[instr['s2']] = True
                    active_regs = [instr['s2']]
                elif opcode_split[1] == 'cm':
                    s1, s2 = reg_table[instr['s1']], const_table[instr['s2']]
                    utilised_table[instr['s1']] = True
                    active_regs = [instr['s1']]
                else:
                    raise Exception("MONARCH - Emulator: Unrecognised opcode '{}'".format(opcode_split[1]))
            
            # Perform computation specified by the opcode
            if opcode == 'nop':
                dly_ctr = instr['s2']
                pc += 1
            elif opcode == 'halt':
                run = False
            else:
                dest_reg = instr['dest']
                if opcode == 'mult':
                    res = s1 * s2
                elif opcode == 'add':
                    res = s1 + s2
                elif opcode == 'sub':
                    res = s1 - s2
                elif opcode == 'div':
                    res = s1 / s2
                elif opcode == 'lut':
                    print("WARNING: Only exponential function is implemented")
                    res = np.exp(s2)
                else:
                    raise Exception("MONARCH - Emulator: Unrecognised opcode {} ".format(opcode))
            
                # Lookup the delay for the operation from arch_dbs
                dly = arch_ops[opcode]['delay']
                
                # Add result to delay pipeline
                dly_res.append({
                    "res": res,
                    "dest": dest_reg,
                    "dly": dly
                })

                pc += 1
        else:
            dly_ctr -= 1
        
        # Extract valid results from delay pipeline and remove them
        new_res = []
        for dlyed_res in dly_res:
            if dlyed_res['dly'] <= 0:
                reg_table[dlyed_res["dest"]] = dlyed_res["res"]

                if utilised_table[dlyed_res["dest"]] == False:
                    print("WARNING: Result at r{} is unused and being overwritten".format(dlyed_res["dest"]))

                # Detect potential race conditions
                if opcode != 'nop' and opcode != 'halt':
                    if dlyed_res["dest"] in active_regs:
                        print("WARNING: Potential race condition: r{}".format(dlyed_res["dest"]))
            else: 
                # Decrement delay counters
                dlyed_res['dly'] -= 1
                new_res.append(dlyed_res)
        
        dly_res = new_res

        if opcode == 'nop':
            print("CYCLE {}: Instr {}".format(clk, opcode))
        else:   
            print("CYCLE {}: Instr {} d1 {} d2 {} res {} dest {}".format(clk, opcode, s1, s2, res, dest_reg))
        clk += 1

    print("TERMINAL REGISTER MAP:")
    for i, dat in enumerate(reg_table):
        print("r{}  {}".format(i, dat))


def emulate_core():
    # Emulates a single core of monarch assembly code.
    # Uses the data in the cache to run a simulation.

    print("Emulating system. NB Floating point format is assumed...")

    asm_filename = "monarch/cache/exec_core0.mem"
    const_filename = "monarch/cache/TEST_CONSTS.mem"
    regrefs_filename = "monarch/cache/rd_regrefs_bank0.mem"
    memfile_filename = "monarch/cache/memfile_bank0_col0.mem"

    script_dir = os.path.dirname(__file__)
    rel_path = "../parsers/arch_dbs.json"
    abs_file_path = os.path.join(script_dir, rel_path)
    
    with open(abs_file_path) as file:
        dbs = json.loads(file.read())
        arch_dbs = dbs

    masm = parse_machcode(asm_filename, arch_dbs['isa'])

    # Build the constant and register tables.
    const_table = parse_consts(
        const_filename, 
        arch_dbs["sys_params"]["datapath_mantissa"], 
        arch_dbs["sys_params"]["datapath_exponent"]
    )

    reg_table = parse_regs(
        regrefs_filename,
        memfile_filename, 
        arch_dbs["sys_params"]["datapath_mantissa"], 
        arch_dbs["sys_params"]["datapath_exponent"],
        arch_dbs["manycore_params"]["working_regs"] + arch_dbs["manycore_params"]["output_regs"]
    )
    
    # Execute code.
    execute_masm(masm, reg_table, const_table, arch_dbs['opcodes'])