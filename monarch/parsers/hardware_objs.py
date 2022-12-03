

class GraphUnit:

    def __init__(self, conn_mat, dly_mat, source_nodes, sink_nodes):

        self.conn_mat = conn_mat
        self.dly_mat = dly_mat
        self.source_nodes = source_nodes
        self.sink_nodes = sink_nodes

        self.arch_dbs = None
