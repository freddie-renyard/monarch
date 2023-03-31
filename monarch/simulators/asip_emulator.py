import os
import json
from parsers.float_compiler import BinCompiler

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

def execute_masm(masm, reg_table, const_table):

    clk = 0
    run = True
    pc = 0

    delay_res = []
    while run:

        instr = masm[pc]
        
        # Decode the instruction
        
        clk += 1

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
    execute_masm(masm, reg_table, const_table)