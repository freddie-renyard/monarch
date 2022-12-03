import numpy as np
from parsers.report_utils import plot_mat

class GraphUnit:

    def __init__(self, conn_mat, dly_mat, source_nodes, sink_nodes):

        self.conn_mat = conn_mat
        self.dly_mat = dly_mat
        self.source_nodes = source_nodes
        self.sink_nodes = sink_nodes

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
            
            non_zero_els = row[np.nonzero(row)]
            if len(non_zero_els):
                min_val = np.min(non_zero_els)
                self.predelays[i] = min_val
                
                mask = np.zeros(np.shape(self.dly_mat)[1])
                mask[np.nonzero(row)] = min_val

                self.dly_mat[i, :] = np.subtract(self.dly_mat[i, :], mask)