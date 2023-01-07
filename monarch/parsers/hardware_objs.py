import numpy as np
from parsers.report_utils import plot_mat
from parsers.bin_compiler import convert_to_hex, convert_to_fixed
from parsers.asm_compiler import allocate_core_instr
from sympy import Symbol
import json
import os

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

                if equal_op and equal_conn and dissimilar and no_square:
                    # The sink nodes compute the same result, combine them
                    self.conn_mat = self.combine_mat_dat(self.conn_mat, sink_node_0, sink_node_1)
                    self.dly_mat = self.combine_mat_dat(self.dly_mat, sink_node_0, sink_node_1)
                    self.remove_nodes([sink_node_1])
                    return True

        # Return false if no optimisations have been found or performed.
        return False

    def remove_dup_branches(self):
        # Analyses the full system CFG in matrix form and removes branches
        # that are duplicates.

        opt = True
        passes = 0
        while opt:
            opt = self.find_and_modify_similars()
            passes += 1
        
        print("MONARCH - Matrix branch optimisation complete. Cycles: {}".format(passes-1))
        print("MONARCH - Post optimisation utilisation:")
        self.report_utilisation()

    def verify_against_dbs(self):
        # Verify that each node has the appropriate number of inputs
        # after graph compilation and optimisation. 
        for j, output_node in enumerate(self.sink_nodes):
            name = str(output_node).split('_')[0]
            if name in self.arch_dbs.keys():
                conns = self.conn_mat[:, j] > 0
                if np.sum(conns) != self.arch_dbs[name]['input_num']:
                    raise Exception("MONARCH - Node {} has {} input connections, rather than {}".format(output_node, np.sum(conns), self.arch_dbs[name]['input_num'] ))

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

        self.cores = dbs['manycore_params']['cores']
        self.input_regs = dbs['manycore_params']['input_regs']
        self.work_regs = dbs['manycore_params']['working_regs']

        self.compile_instrs()

    def compile_instrs(self):
        
        input_num = len([x for x in self.graph_unit.source_nodes if type(x) != str])
        if input_num > self.input_regs:
            raise Exception("MONARCH - Too many input registers are needed to realise the system.")
        
        output_num = len([x for x in self.graph_unit.sink_nodes if type(x) != str])
        if output_num > self.work_regs:
            raise Exception("MONARCH - Too many input registers are needed to realise the system.")

        instrs = [[] for _ in range(self.cores)]
        core_i = 0
        
        
        # Build the initial register map
        reg_map = []
        output_n = len([x for x in self.graph_unit.sink_nodes if type(x) != str])
        for i in range(self.work_regs):
            if i < output_n:
                reg_map.append('output_{}'.format(i))
            else:
                reg_map.append(None)
        
        completed = []
        for i in range(self.cores):
            
            op, completed = allocate_core_instr(
                self.graph_unit.conn_mat, 
                self.graph_unit.source_nodes, 
                self.graph_unit.sink_nodes,
                completed
            )

            instrs[core_i].append(op)

        print(instrs, completed)
            
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

            with open("monarch/cache/{}_{}_state".format(self.name, key), "w+") as file:
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

        with open("monarch/cache/{}.vh".format(filename), "w+") as output_file:
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
        with open("monarch/cache/{}.vh".format(filename), "w+") as output_file:
            output_file.write(vh_str)