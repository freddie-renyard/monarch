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
                    dup_i = self.source_nodes.index(node)
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