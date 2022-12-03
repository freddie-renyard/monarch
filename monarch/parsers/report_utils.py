from matplotlib import pyplot as plt
import numpy as np

def stringify_lst(lst):

    for i, el in enumerate(lst):
        if type(el) != str:
            lst[i] = str(el)

    return lst

def plot_mat(mats, source_nodes, sink_nodes, mat_titles=[]):

    # stringify source and sink nodes
    source_nodes = stringify_lst(source_nodes)
    sink_nodes = stringify_lst(sink_nodes)

    fig, axs = plt.subplots(1, len(mats))
    for val, (mat, mat_title) in enumerate(zip(mats, mat_titles)):
        
        axs[val].matshow(mat, interpolation='nearest', cmap='seismic')
        axs[val].set_title(mat_title)

        xaxis = np.arange(len(sink_nodes))
        yaxis = np.arange(len(source_nodes))

        # Add catch case for predelays.
        if mat_title != "Predelays":
            axs[val].set_xticks(xaxis)
            axs[val].set_xticklabels(sink_nodes, rotation=90)

        axs[val].set_yticks(yaxis)
        axs[val].set_yticklabels(source_nodes)

        for (i, j), z in np.ndenumerate(mat):
            axs[val].text(j, i, '{}'.format(int(z)), ha='center', va='center')

    plt.show()

    