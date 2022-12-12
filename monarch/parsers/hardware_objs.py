import numpy as np
from parsers.report_utils import plot_mat
from sympy import Symbol

class GraphUnit:

    def __init__(self, conn_mat, dly_mat, source_nodes, sink_nodes, assoc_dat):

        self.conn_mat = conn_mat
        self.dly_mat = dly_mat
        self.source_nodes = source_nodes
        self.sink_nodes = sink_nodes
        self.assoc_dat = assoc_dat

        self.arch_dbs = None
        self.predelays = []

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

                if equal_op and equal_conn and dissimilar:
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
        
        print("MONARCH: Matrix branch optimisation complete. Cycles: {}".format(passes))

    def verify_against_dbs(self):
        # Verify that each node has the appropriate number of inputs
        # after graph compilation and optimisation. 
        for j, output_node in enumerate(self.sink_nodes):
            name = str(output_node).split('_')[0]
            if name in self.arch_dbs.keys():
                conns = self.conn_mat[:, j] > 0
                if np.sum(conns) != self.arch_dbs[name]['input_num']:
                    raise Exception("MONARCH - Node {} has {} input connections, rather than {}".format(output_node, np.sum(conns), self.arch_dbs[name]['input_num'] ))
