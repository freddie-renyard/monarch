from asyncio.sslproto import constants
import numpy as np
from parsers.report_utils import plot_mat
from parsers.bin_compiler import convert_to_hex, convert_to_fixed
from parsers.asm_compiler import allocate_core_instr, update_reg_map, update_clk_cycle, disp_exec_thread, find_terminal_instrs, find_stale_results, instr_to_asm, collapse_nops
from parsers.asm_compiler import instr_to_machcode, preprocess_asm
from parsers.lut_compiler import generate_lut
from sympy import Symbol, sympify, Float
import json
from copy import copy
import os

cache_path = "monarch/cache"

class GraphUnit:

    def __init__(self, conn_mat, dly_mat, source_nodes, sink_nodes, assoc_dat, pipeline_depth):

        self.conn_mat = conn_mat
        self.dly_mat = dly_mat
        self.source_nodes = source_nodes
        self.sink_nodes = sink_nodes
        self.assoc_dat = assoc_dat
        self.pipeline_depth = pipeline_depth

        self.predelays = []

        # Open architecture database.
        script_dir = os.path.dirname(__file__)
        rel_path = "arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            self.arch_dbs = dbs['opcodes']

        self.combine_vars()
        self.reorder_source_nodes()
        self.reorder_sink_nodes()

        self.remove_dup_branches()

        self.sort_matrices()

        self.verify_against_dbs()
        self.compute_predelay()

    def show_report(self):
        plot_mat(
            [self.conn_mat, np.expand_dims(self.predelays, axis=1), self.dly_mat],
            self.source_nodes,
            self.sink_nodes,
            ["Connectivity", "Predelays", "Delay"]
        )

    def combine_vars(self):
        # Combines the variables of the trees together and removes duplicates from source node list.

        duplicates = len(self.source_nodes) > len(set(self.source_nodes)) 
        while duplicates:
            for node in self.source_nodes:
                if self.source_nodes.count(node) > 1:
                    
                    # Take the lowest index occurance.
                    dup_i      = self.source_nodes.index(node)
                    line_1     = self.conn_mat[dup_i, :]
                    line_1_dly = self.dly_mat[dup_i, :]
                    del self.source_nodes[dup_i]
                    self.conn_mat = np.delete(self.conn_mat, (dup_i), axis=0)
                    self.dly_mat = np.delete(self.dly_mat, (dup_i), axis=0)

                    dup_i = self.source_nodes.index(node)
                    line_2     = self.conn_mat[dup_i, :]
                    line_2_dly = self.dly_mat[dup_i, :]
                    self.conn_mat[dup_i, :] = np.add(line_1, line_2)
                    self.dly_mat[dup_i, :] = np.add(line_1_dly, line_2_dly)

                    break

            duplicates = len(self.source_nodes) > len(set(self.source_nodes)) 

    def compute_predelay(self):
        # Adds predelay registers to the unit.

        self.predelays = np.zeros(np.shape(self.dly_mat)[0])
        
        for i, row in enumerate(self.dly_mat):
            
            non_zero_els = row[np.nonzero(self.conn_mat[i, :])]
            if len(non_zero_els):
                min_val = np.min(non_zero_els)
                self.predelays[i] = min_val
                
                mask = np.zeros(np.shape(self.dly_mat)[1])
                mask[np.nonzero(row)] = min_val

                self.dly_mat[i, :] = np.subtract(self.dly_mat[i, :], mask)

    def reorder_source_nodes(self):
        # Reorder the source nodes by the architecture database.

        symbols = [int(type(node) == str) for node in self.source_nodes]
        symbols = np.array(symbols)

        for i, op in enumerate(self.arch_dbs):
            temp = np.array([int(str(node).find(op) > -1) for node in self.source_nodes])
            temp *= 1 + i
            symbols = np.add(symbols, temp)

        sort_is = np.argsort(symbols)

        self.source_nodes = np.array(self.source_nodes)[sort_is]
        self.conn_mat = self.conn_mat[sort_is, :]
        self.dly_mat = self.dly_mat[sort_is, :]

    def reorder_sink_nodes(self):
        # Reorder the sink nodes by the architecture database.
        # TODO Combine with the code above.

        symbols = [int(type(node) == str) for node in self.sink_nodes]
        symbols = np.array(symbols)

        for i, op in enumerate(self.arch_dbs):
            temp = np.array([int(str(node).find(op) > -1) for node in self.sink_nodes])
            temp *= 1 + i
            symbols = np.add(symbols, temp)

        sort_is = np.argsort(symbols)

        self.sink_nodes = np.array(self.sink_nodes)[sort_is]
        self.conn_mat = self.conn_mat[:, sort_is]
        self.dly_mat = self.dly_mat[:, sort_is]

    def compute_delay_regs(self):
        total = int(np.sum(self.dly_mat))
        total += int(np.sum(self.predelays))
        return total

    def compute_max_depth(self):
        pass

    def combine_mat_dat(self, mat, node_0, node_1):
        # Combines relevant matrix data and deletes depreciated rows.

        node_0_source_i = np.where(self.source_nodes == node_0)[0]
        node_1_source_i = np.where(self.source_nodes == node_1)[0]
        node_0_sink_j = np.where(self.sink_nodes == node_0)[0]
        node_1_sink_j = np.where(self.sink_nodes == node_1)[0]

        node_0_col = mat[:, node_0_sink_j]
        node_1_col = mat[:, node_1_sink_j]

        # Verify that combination condition is met - Throw exception because there is no hardware
        # provision for meeting unequal node delays. Implementing this is possible by using the pre-delay registers.
        if not np.array_equal(node_0_col, node_1_col):
            raise Exception("MONARCH - There is assymmetric input node delay in a branch that is attempting to be merged.")
        
        # Combine matrix rows, adding all data to lower indexed row/column.
        node_0_row = mat[node_0_source_i, :]
        node_1_row = mat[node_1_source_i, :]
        mat[node_0_source_i, :] = np.add(node_0_row, node_1_row)

        # Delete depreciated rows.
        mat = np.delete(mat, [node_1_sink_j], axis=1)
        mat = np.delete(mat, [node_1_source_i], axis=0)

        return mat

    def remove_nodes(self, node_lst):
        # Deletes nodes by index.

        for node in node_lst:
            node_source_i = np.where(self.source_nodes == node)[0]
            node_sink_j = np.where(self.sink_nodes == node)[0]

            self.source_nodes = np.delete(self.source_nodes, [node_source_i], axis=0)
            self.sink_nodes = np.delete(self.sink_nodes, [node_sink_j], axis=0)
    
    def find_and_modify_similars(self):
        # Exhaustively check for sink node symmetry.

        for j_0, sink_node_0 in enumerate(self.sink_nodes):
            for j_1, sink_node_1 in enumerate(self.sink_nodes):
                # Strip nodes of their unique IDs
                target_nodes = [sink_node_0, sink_node_1]
                target_ops = [str(x).split("_")[0] for x in target_nodes]

                # Fetch the sink nodes respective connectivity vectors
                target_rows = [self.conn_mat[:, j_0], self.conn_mat[:, j_1]]

                # Set up target conditions
                equal_op = target_ops[0] == target_ops[1]
                equal_conn = np.array_equal(target_rows[0], target_rows[1])
                dissimilar = str(sink_node_0) != str(sink_node_1)
                
                no_square = not (("square" in str(sink_node_0)) and ("square" in str(sink_node_1)))

                if equal_op and equal_conn and dissimilar:
                    # The sink nodes compute the same result, combine them
                    self.conn_mat = self.combine_mat_dat(self.conn_mat, sink_node_0, sink_node_1)
                    self.dly_mat = self.combine_mat_dat(self.dly_mat, sink_node_0, sink_node_1)
                    self.remove_nodes([sink_node_1])
                    return True

        # Return false if no optimisations have been found or performed.
        return False

    def rectify_error(self, err_node):

        op = err_node.split("_")[0]
        op_index = err_node.split("_")[1]
        
        if op == 'mult':
            sink_i = np.where(self.sink_nodes == err_node)[0][0]
            source_i = np.where(self.source_nodes == err_node)[0][0]
            
            new_op = "square_" + op_index
            self.source_nodes[source_i] = new_op
            self.sink_nodes[sink_i] = new_op

        self.verify_against_dbs()

    def remove_dup_branches(self):
        # Analyses the full system CFG in matrix form and removes branches
        # that are duplicates.

        opt = True
        passes = 0
        while opt:
            opt = self.find_and_modify_similars()
            err_node = self.verify_against_dbs(throw_exception=False)
            if err_node:
                self.rectify_error(err_node)
            
            passes += 1
        
        print("MONARCH - Matrix branch optimisation complete. Cycles: {}".format(passes-1))
        print("MONARCH - Post optimisation utilisation:")
        self.report_utilisation()

    def verify_against_dbs(self, throw_exception=True):
        # Verify that each node has the appropriate number of inputs
        # after graph compilation and optimisation. 
        for j, output_node in enumerate(self.sink_nodes):
            name = str(output_node).split('_')[0]
            if name in self.arch_dbs.keys():
                conns = self.conn_mat[:, j] > 0
                if np.sum(conns) != self.arch_dbs[name]['input_num']:
                    if throw_exception:
                        raise Exception("MONARCH - Node {} has {} input connections, rather than {}".format(output_node, np.sum(conns), self.arch_dbs[name]['input_num'] ))
                    else:
                        return output_node

        return None
    
    def report_utilisation(self):

        for op in self.arch_dbs:
            blocks = [node for node in self.source_nodes if type(node) == str]
            blocks = [node for node in blocks if node.find(op) > -1]
            print("\t{} : {} blocks".format(self.arch_dbs[op]["block_name"], len(blocks)))

        print("\tPipeline depth: {} registers".format(self.pipeline_depth))
        print("\tTotal delay register count: {} registers\n".format(self.compute_delay_regs()))

    def sort_matrices(self):
        # Sorts the matrices and source/sink nodes into ascending numerical order
        # to make hardware routing easier.

        sort_lst = []
        for el in self.source_nodes:
            if type(el) == str:
                sort_lst.append(int(el.split("_")[1]))
            else:
                sort_lst.append(-1)
        
        sorted_is = np.argsort(sort_lst)

        self.source_nodes = self.source_nodes[sorted_is]        
        self.conn_mat     = self.conn_mat[sorted_is, :]
        self.dly_mat      = self.dly_mat[sorted_is, :]

        sort_lst = []
        for el in self.sink_nodes:
            if type(el) == str:
                sort_lst.append(int(el.split("_")[1]))
            else:
                sort_lst.append(-1)
        
        sorted_is = np.argsort(sort_lst)

        self.sink_nodes = self.sink_nodes[sorted_is]        
        self.conn_mat   = self.conn_mat[:, sorted_is]
        self.dly_mat    = self.dly_mat[:, sorted_is]

class ManycoreUnit:

    def __init__(self, target_unit, args, init_state, dt, name="manycore_1"):

        self.graph_unit = target_unit
        self.args = {**args, **init_state, "dt": dt}
        self.name = name

        # Open architecture database.
        script_dir = os.path.dirname(__file__)
        rel_path = "arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            self.arch_dbs = dbs

        self.cores       = dbs['manycore_params']['cores']
        self.work_regs   = dbs['manycore_params']['working_regs']
        self.output_regs = dbs['manycore_params']['output_regs']

        # Determine the inputs and the constants.
        # First type of constants are inferred from the equations by the tree compiler.
        consts = {}
        for node in self.graph_unit.source_nodes:
            if type(node) != str and 'const' in str(node):
                # Extract the value of the constant.
                num = str(node).split("_")[0]
                real_num = Float(sympify(num))
                consts[str(node)] = real_num

        # Second type of constants are specified by the user/inferred from context
        # i.e. dt will almost always be a constant.
        consts = {
            **consts, 
            "dt": dt
        }

        # Convert the constants to a list to ensure no reordering during compilation.
        self.consts = []
        self.const_names = []
        for key in consts:
            self.consts.append(consts[key]) 
            self.const_names.append(key) 

        self.clear_cache()

        asm = self.compile_instrs()

        self.report_exec_time(asm)
        self.asm_to_machcode(asm)

    def compile_instrs(self, verbose=False):
        
        input_nodes = [str(x) for x in self.graph_unit.source_nodes if type(x) != str]
        input_nodes = [str(x) for x in input_nodes if x not in self.const_names]
        input_num = len(input_nodes)
        if input_num > self.work_regs:
            raise Exception("MONARCH - Too many input registers ({} registers) are needed to realise the system.".format(input_num))
        
        output_num = len([x for x in self.graph_unit.sink_nodes if type(x) != str])
        if output_num > self.work_regs:
            raise Exception("MONARCH - Too many input registers are needed to realise the system.")

        instrs = [[] for _ in range(self.cores)]
        asm    = [[] for _ in range(self.cores)]

        # Determine the output instructions.
        terminal_instrs, terminal_vars = find_terminal_instrs(
            self.graph_unit.conn_mat, 
            self.graph_unit.source_nodes,
            self.graph_unit.sink_nodes
        )

        # Determine the seed list for the instruction allocation.
        # TODO Fix the allocation code, as it causes the compiler to hang in some instances.
        """
        primaries = determine_primary(
            self.graph_unit.conn_mat, 
            self.graph_unit.source_nodes,
            self.graph_unit.sink_nodes,
            target_op='div'
        )
        """
        primaries = []

        # Initialise the register map. 
        reg_map = []
        self.start_locs = {}
        self.end_locs = {}
        for i in range(self.work_regs + self.output_regs):
            if i < len(input_nodes):
                reg_map.append({
                    "d": input_nodes[i],
                    "s": "valid"
                })
                self.start_locs[input_nodes[i]] = i
            else:
                reg_map.append({
                    "d": None,
                    "s": "avail"
                })
        
        completed = []
        completed_outs = 0

        while completed_outs != len(terminal_instrs):
            
            # Update the registers.
            reg_map, completed_outs = update_clk_cycle(reg_map, 
                terminal_instrs, 
                completed_outs
            )

            # Determine which instruction results in the register map are 
            # stale and can be overwritten with a new instruction.
            reg_map = find_stale_results(
                reg_map,
                self.graph_unit.conn_mat,
                self.graph_unit.source_nodes,
                self.graph_unit.sink_nodes,
                completed
            )

            new_instrs = []

            """
            # Add each instruction to the register map.
            for i, reg in enumerate(reg_map):
                print(i, reg)
            print()
            input()
            """

            # Allocate an instruction to each core.
            for i in range(self.cores):
                op, completed = allocate_core_instr(
                    self.graph_unit.conn_mat, 
                    self.graph_unit.source_nodes, 
                    self.graph_unit.sink_nodes,
                    reg_map,
                    self.graph_unit.assoc_dat,
                    self.const_names,
                    self.arch_dbs,
                    primaries,
                    completed
                )

                new_instrs.append(op)

            for core_instr in new_instrs:
                reg_map, self.end_locs = update_reg_map(
                    reg_map, 
                    core_instr, 
                    terminal_instrs,
                    terminal_vars,
                    self.work_regs, 
                    self.arch_dbs,
                    self.end_locs
                )
            
            # Append the instructions to the core instruction threads
            for i, new_instr in enumerate(new_instrs):
                instrs[i].append(new_instr)

            # Append the instructions to the main assembly thread, which
            # will be compiled to machine code.
            for i, new_instr in enumerate(new_instrs):
                asm_instr = instr_to_asm(new_instr, reg_map, self.const_names)
                asm[i].append(asm_instr)

        if verbose:
            for i, instr_asm in enumerate(instrs):
                print("// MONARCH - CORE {} ASSEMBLY".format(i))
                disp_exec_thread([instr_asm])
                print()

        return asm

    def asm_to_machcode(self, asm, verbose=True):
        # Compile threads of assembly into machine code, as per the manycore MONArch ISA.

        # Perform nop collapse to compress memory footprint.
        asm = [collapse_nops(x) for x in asm]

        # Add a halt instruction to the end of every core program.
        asm = [x + [['halt', None, None, None]] for x in asm]

        # Preprocess the assembly to include hardware-specific instruction variants,
        # such as use of constants in computation.
        asm = [preprocess_asm(core_asm) for core_asm in asm]

        if verbose:
            for index, core_asm in enumerate(asm):
                print("// MONARCH CORE {} FINAL ASSEMBLY".format(index))
                disp_exec_thread([core_asm])
            
        for i, core_asm in enumerate(asm):
            with open(os.path.join(cache_path, "exec_core{}.mem".format(i)), "w+") as file:
                machcode_bin = '// Core {} machine code for tile {} \n'.format(i, self.name)

                for instr in core_asm:
                    machcode_bin += instr_to_machcode(instr, self.arch_dbs)
                
                file.write(machcode_bin)

    def report_exec_time(self, asm):
        
        lengths = []
        nop_counts = []
        for core_asm in asm:
            lengths.append(len(core_asm))

            nop_count = 0
            for instr in core_asm:
                if instr[0] == 'nop':
                    nop_count += 1
            nop_counts.append(nop_count)

        max_len = max(lengths)
        print("MONARCH - Multicore execution time: {} cycles".format(max_len + 1))
        for i, counts in enumerate(zip(lengths, nop_counts)):
            print("MONARCH - NOP proportion for core {}: {:.1f}%".format(i, 100.0 * float(counts[1]) / float(counts[0])))

    def clear_cache(self):
        # Clear the cache before writing new data.
        for f in os.listdir(cache_path):
            os.remove(os.path.join(cache_path, f))

class HardwareUnit:

    def __init__(self, target_unit, args, init_state, dt, name="unit_1"):

        self.graph_unit = target_unit
        self.args = {**args, **init_state, "dt": dt}
        self.name = name

        # Open architecture database.
        script_dir = os.path.dirname(__file__)
        rel_path = "arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            self.arch_dbs = dbs

        self.compile_to_header("gu_params")
        self.compile_model_inst_vars()

    def lst_to_str(self, target_lst, newlines=False):
        
        output_str = ""
        for i, el in enumerate(target_lst):
            output_str += str(int(el)) + ", "

            if i == len(target_lst) - 1:
                output_str = output_str[:-2]
                
            if newlines:
                output_str += "\n    "
        
        return output_str

    def get_source_type(self, target):
        # Returns the integer type of a given node.
        if type(target) == str:
            target_op = target.split("_")[0]
            return self.arch_dbs["opcodes"][target_op]["op_index"]
        else:
            return 0

    def compile_model_inst_vars(self):
        # Compiles the variables of a model into their binary representation.

        data_width = self.arch_dbs["sys_params"]["datapath_width"]
        radix = self.arch_dbs["sys_params"]["datapath_radix"]
        
        for key in self.args:
            if len([self.args[key]]) == 1:
                output = [convert_to_fixed(self.args[key], data_width, radix)]
            else:
                output = convert_to_fixed(self.args[key], data_width, radix)
            
            output = [convert_to_hex(x) for x in output]

            with open(os.path.join(cache_path, "{}_{}_state".format(self.name, key)), "w+") as file:
                for hex_num in output:
                    file.write(hex_num)
                    file.write("\n")

    def compile_to_header(self, filename):
        # This method compiles a target unit to a verilog header 
        # for pipeline synthesis.

        with open("monarch/dbs/gu_params_template.vh") as vh_file:
            vh_str = vh_file.read()

        # Compile integer constants
        vh_str = vh_str.replace("<source_num>", str(len(self.graph_unit.source_nodes)))
        vh_str = vh_str.replace("<sink_num>",   str(len(self.graph_unit.sink_nodes)))
        vh_str = vh_str.replace("<pipe_depth>", str(self.graph_unit.pipeline_depth))
        vh_str = vh_str.replace("<input_num>",  str(len([x for x in self.graph_unit.source_nodes if type(x) != str])))
        vh_str = vh_str.replace("<output_num>",  str(len([x for x in self.graph_unit.sink_nodes if type(x) != str])))
        
        predelay_str = self.lst_to_str(self.graph_unit.predelays)
        vh_str = vh_str.replace("<arr_predelay>", predelay_str)

        # Compile source node params.   
        sources = [self.get_source_type(x) for x in self.graph_unit.source_nodes]
        source_str = self.lst_to_str(sources)
        vh_str = vh_str.replace("<arr_source_type>", source_str)

        # Compile the connectivity arrays.
        conn_primaries   = []
        dly_primaries    = []
        conn_secondaries = []
        dly_secondaries  = []
        err_val = -1

        for j in range(len(self.graph_unit.sink_nodes)):
            target_col  = self.graph_unit.conn_mat[:, j]
            dly_col     = self.graph_unit.dly_mat[:, j]

            primary = target_col == 1
            secondary = target_col == 2

            primary_i = np.nonzero(primary)[0]
            conn_primaries.append(primary_i)
            dly_primaries.append(dly_col[primary_i])
            
            secondary_i = np.nonzero(secondary)[0]
            if len(secondary_i):
                conn_secondaries.append(secondary_i)
                dly_secondaries.append(dly_col[secondary_i])
            else:
                conn_secondaries.append(err_val)
                dly_secondaries.append(err_val)
        
        vh_str = vh_str.replace("<arr_in_1>", self.lst_to_str(conn_primaries))
        vh_str = vh_str.replace("<arr_in_2>", self.lst_to_str(conn_secondaries))
        vh_str = vh_str.replace("<arr_dly_1>", self.lst_to_str(dly_primaries))
        vh_str = vh_str.replace("<arr_dly_2>", self.lst_to_str(dly_secondaries))

        with open(os.path.join(cache_path, "{}.vh".format(filename)), "w+") as output_file:
            output_file.write(vh_str)

class CFGU:

    def __init__(self):
        # Open architecture database.
        script_dir = os.path.dirname(__file__)
        rel_path = "arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            self.arch_dbs = dbs

        self.compile_gbl_header()
            
    def compile_gbl_header(self):

        with open("monarch/dbs/gbl_params_template.vh") as vh_file:
            vh_str = vh_file.read()

        vh_str = vh_str.replace("<path_width>",  str(self.arch_dbs["sys_params"]["datapath_width"]))
        vh_str = vh_str.replace("<radix_width>",  str(self.arch_dbs["sys_params"]["datapath_radix"]))

        filename = "gbl_params"
        with open(os.path.join(cache_path, "{}.vh".format(filename)), "w+") as output_file:
            output_file.write(vh_str)

class Tile:

    def __init__(self, hardware_unit, instances, sys_data, sys_state_vars, const_names=['dt']):

        self.hardware_unit = hardware_unit
        self.sys_state_vars = list(sys_state_vars)

        self.sys_data = copy(sys_data)
        for key, item in zip(hardware_unit.const_names, hardware_unit.consts):
            self.sys_data.update({key: item})
        
        # Open architecture database.
        script_dir = os.path.dirname(__file__)
        rel_path = "arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            self.arch_dbs = dbs

        self.dpath_radix = dbs["sys_params"]["datapath_radix"]
        self.dpath_width = dbs["sys_params"]["datapath_width"]
        self.reg_width   = dbs["manycore_params"]["machcode_params"]["reg_ptr_width"]
        self.columns     = dbs["manycore_params"]["columns"]
        self.mem_banks   = dbs["manycore_params"]["mem_banks"]

        if instances % self.columns != 0:
            raise Exception("MONARCH - The total number of instances does not divide evenly into the number of columns specified.")

        self.var_names, self.const_names = self.partition_variables(hardware_unit.const_names, list(sys_state_vars))
        print(self.var_names, self.const_names)
        self.resynth_luts()

        self.generate_insts(instances)
        self.compile_consts()

        self.compile_pkg()

    def partition_variables(self, sys_consts, sys_state_vars):
        # Partitions system variables into the groups needed to run systems:
        # 1. State variables (i.e. variables that vary with time)
        # 2. Instance variables (i.e. variables that vary between parallel model instances)
        # 3. Constants

        inst_var_names, const_names, = [], []
        for var in self.sys_data:
            if str(var) in sys_consts:
                const_names.append(var)
            elif str(var) not in sys_state_vars:
                inst_var_names.append(var)
        
        # Ensure that the state variables are first in the list.
        vars = sorted(sys_state_vars)
        vars += sorted(inst_var_names)

        return vars, const_names

    def generate_insts(self, n_insts):
        # Uses the data passed to the object to construct different instances of
        # the model.

        # Allocate each variable to a memory bank.
        bank_names = [[] for _ in range(self.mem_banks)]
        bank_n_rd_cell = [0 for _ in range(self.mem_banks)]
        bank_n_wr_cell = [0 for _ in range(self.mem_banks)]

        for i, var in enumerate(self.var_names):

            if str(var) in self.sys_state_vars:
                bank_n_wr_cell[i % self.mem_banks] += 1

            bank_n_rd_cell[i % self.mem_banks] += 1
            bank_names[i % self.mem_banks].append(var)

        # Allocate the instance data to the banks.
        structured_banks = [[] for _ in range(self.mem_banks)]
        for i in range(n_insts):
            for j, bank in enumerate(bank_names):
                for key in bank:
                    if hasattr(self.sys_data[key], "__iter__"):
                        structured_banks[j].append(self.sys_data[key][i]) 
                    else:
                        structured_banks[j].append(self.sys_data[key]) 
        
        # Structure the data into columns
        final_banks = [[[] for _ in range(self.columns)] for _ in range(self.mem_banks)]
        for i, mem_bank in enumerate(structured_banks):
            for j, dat in enumerate(mem_bank):
                final_banks[i][(j // bank_n_rd_cell[i]) % self.columns].append(dat)
        
        # Output all relevant files.
        for i in range(self.mem_banks):

            # File 1: The control parameters used to determine the memory termination, etc.
            with open(os.path.join(cache_path, "ctrl_params_bank{}.mem".format(i)), "w+") as file:
                total_size = bank_n_rd_cell[i] * ((n_insts // self.columns) - 1)
                
                file.write(convert_to_fixed(total_size, 16, 0) + '\n')
                file.write(convert_to_fixed(bank_n_rd_cell[i]-1, 16, 0) + '\n')
                file.write(convert_to_fixed(bank_n_wr_cell[i]-1, 16, 0) + '\n')

            # File 2: The actual memory files.
            for bank_i, mem_bank in enumerate(final_banks):
                for col_i, mem_col in enumerate(mem_bank):
                    with open(os.path.join(cache_path, "memfile_bank{}_col{}.mem".format(bank_i, col_i)), "w+") as file:
                        for dat in mem_col:
                            file.write(convert_to_fixed(dat, self.dpath_width, self.dpath_radix) + '\n')

            # File 3: The register reference files.
            for bank_i, name_set in enumerate(bank_names):
                rd_reg_file = open(os.path.join(cache_path, "rd_regrefs_bank{}.mem".format(bank_i)), "w+")
                wr_reg_file = open(os.path.join(cache_path, "wr_regrefs_bank{}.mem".format(bank_i)), "w+")

                for name in name_set:
                    if name in list(self.hardware_unit.start_locs.keys()):
                        target = self.hardware_unit.start_locs[name]
                    elif (str(name) + "_pre") in list(self.hardware_unit.start_locs.keys()):
                        
                        target = self.hardware_unit.start_locs[str(name) + "_pre"]
                        wr_target = self.hardware_unit.end_locs[name + "_post"]

                        wr_reg_file.write(convert_to_fixed(wr_target, self.reg_width, 0, signed=False) + '\n')

                    rd_reg_file.write(convert_to_fixed(target, self.reg_width, 0, signed=False) + '\n')

                rd_reg_file.close()
                wr_reg_file.close()

    def compile_consts(self):
        with open(os.path.join(cache_path, "TEST_CONSTS.mem"), "w+") as file:
            for name in self.const_names:
                file.write(convert_to_fixed(self.sys_data[name], self.dpath_width, self.dpath_radix) + '\n')

    def compile_pkg(self):
        # Compiles the package file for the tile hardware. Used for FPGA implementations where
        # the hardware is not static.

        with open("monarch/dbs/tile_pkg_template.sv") as sv_file:
            sv_str = sv_file.read()

        sv_str = sv_str.replace("<instr_width>", str(self.arch_dbs["manycore_params"]["machcode_params"]['instr_width']))
        sv_str = sv_str.replace("<reg_width>", str(self.arch_dbs["manycore_params"]["machcode_params"]['reg_ptr_width']))
        sv_str = sv_str.replace("<data_width>", str(self.arch_dbs["sys_params"]["datapath_width"]))
        sv_str = sv_str.replace("<data_radix>", str(self.arch_dbs["sys_params"]["datapath_radix"]))
        sv_str = sv_str.replace("<n_cores>", str(self.arch_dbs["manycore_params"]["cores"]))
        sv_str = sv_str.replace("<n_columns>", str(self.arch_dbs["manycore_params"]["columns"]))
        sv_str = sv_str.replace("<n_bank>", str(self.arch_dbs["manycore_params"]["mem_banks"]))
        sv_str = sv_str.replace("<n_bank_size>", str(self.arch_dbs["manycore_params"]["mem_bank_size"]))

        # TODO refactor these parameters out of the RTL
        sv_str = sv_str.replace("<n_ext_rd_ports>", str(self.arch_dbs["manycore_params"]["mem_banks"]))
        sv_str = sv_str.replace("<n_ext_wr_ports>", str(self.arch_dbs["manycore_params"]["mem_banks"]))
        
        with open(os.path.join(cache_path, "tile_pkg.sv"), "w+") as file:
            file.write(sv_str)
    
    def resynth_luts(self):
        # Resynthesises the lookup tables to ensure that they are up to date with current system parameters.
        generate_lut("e")