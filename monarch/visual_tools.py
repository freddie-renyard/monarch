import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from math import log10, log2

def plot_2d_phase_space(phase_object, name="", show_arrows=True):
    """Plots a 2D phase diagram for a given PhaseSpace object and shows
    the output as a quiver plot superimposed on a image plot of the total
    gradient vector magnitude.
    """

    if phase_object.dimensions != 2:
        raise ValueError("Dimensionality of the input PhaseSpace Object is not 2.")

    # Compute the magnitude of each vector.
    x_deltas = np.rot90(phase_object.phase_space[:, :, 0])
    y_deltas = np.rot90(phase_object.phase_space[:, :, 1])
    magnitudes = np.sqrt((x_deltas**2 + y_deltas**2))

    skip = phase_object.resolution // 32
    xgrid = range(0,phase_object.resolution)
    ygrid = range(0,phase_object.resolution)
    x,y = np.meshgrid(xgrid,ygrid)

    # Determine plot name
    plot_name = "2D phase plot"
    if name != "":
        plot_name += " of " + name

    plt.title(plot_name)
    plt.imshow(magnitudes)
    if show_arrows:
        plt.quiver(x[::skip,::skip], y[::skip,::skip],
                np.rot90(phase_object.phase_space[::skip,::skip, 0]),
                np.rot90(phase_object.phase_space[::skip,::skip, 1]),
                color='r',
                scale=1
        )
    
    plt.axis('off')
    plt.show()

def plot_3d_phase_space(phase_object, name=""):
    """Plots a 3D quiver plot which corresponds to the phase space
    information in the passed PhaseSpace object.
    """

    if phase_object.dimensions != 3:
        raise ValueError("Dimensionality of the input PhaseSpace Object is not 3.")

    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(0, phase_object.resolution),
                        np.arange(0, phase_object.resolution),
                        np.arange(0, phase_object.resolution))

    # Determine plot name
    plot_name = "3D phase plot"
    if name != "":
        plot_name += " of " + name

    plt.title(plot_name)
    
    skip = phase_object.resolution // 16
    ax.quiver(
        x[::skip,::skip,::skip], 
        y[::skip,::skip,::skip], 
        z[::skip,::skip,::skip], 

        phase_object.phase_space[0,::skip,::skip,::skip], 
        phase_object.phase_space[1,::skip,::skip,::skip], 
        phase_object.phase_space[2,::skip,::skip,::skip], 

        length=0.7,
        color="r",
        normalize=True
    )

    plt.show()

def plot_2d_components(phase_object):
    """Plots the dimension components of the velocity vectors of a compiled
    phase space.
    """

    x_deltas = np.rot90(phase_object.phase_space[:, :, 0])
    y_deltas = np.rot90(phase_object.phase_space[:, :, 1])

    # Compute absolute maximum value of vector components
    x_max = np.abs(x_deltas).max()
    y_max = np.abs(y_deltas).max()
    
    # Normalise the components of the vectors.
    x_deltas = x_deltas / x_max
    y_deltas = y_deltas / y_max

    # Divide by a further factor of 2 to ensure the full 
    # range of the components of the vectors is 1.
    x_deltas = x_deltas / 2.0
    y_deltas = y_deltas / 2.0

    # Offset the components of the vectors.
    x_deltas += 0.5
    y_deltas += 0.5

    plt.subplot(1,2,1) 
    plt.title("X components of input vector")
    plt.imshow(x_deltas, cmap='seismic')
    plt.subplot(1,2,2)
    plt.title("Y components of input vector")
    plt.imshow(y_deltas, cmap='seismic')
    plt.show()

def plot_histogram(phase_object, bins=100):
    """Plots a histogram of all the values of all the vectors in phase space.
    Used for assessing full dynamic range of a compiled system.
    """

    flattened_space = phase_object.phase_space.flatten()

    histogram, bin_edges = np.histogram(flattened_space, bins=bins)

    # Compute the dynamic range of all the values.
    abs_vals = np.abs(flattened_space)
    max_val = abs_vals.max()

    # Ensure that zero isn't chosen as a minimum value to prevent division by zero.
    abs_vals = abs_vals[abs_vals != 0.0]
    min_val = abs_vals.min()

    print(max_val, min_val)
    dyn_range = log2(max_val / min_val)

    # Ensure the bar width is equal.
    bar_width = bin_edges[1]-bin_edges[0] 

    plt.bar(bin_edges[:-1], histogram, width=bar_width, align='edge')
    plt.title('Histogram of Phase Space Values. Dynamic range: {} bits'.format(int(dyn_range)))
    plt.show()