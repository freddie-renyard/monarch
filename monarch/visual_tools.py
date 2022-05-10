from cmath import phase
from unicodedata import name
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from math import log10, log2, sqrt

def plot_2d_phase_space(phase_object, name="", show_arrows=True, show_fig=False):
    """Plots a 2D phase diagram for a given PhaseSpace object and shows
    the output as a quiver plot superimposed on a image plot of the total
    gradient vector magnitude.
    """

    plot_space_x = phase_object.phase_space[:, :, 0]
    plot_space_y = phase_object.phase_space[:, :, 1]

    if phase_object.dimensions != 2:
        raise ValueError("Dimensionality of the input PhaseSpace Object is not 2.")

    # Compute the magnitude of each vector.
    magnitudes = np.sqrt((plot_space_x**2 + plot_space_y**2))

    skip = phase_object.resolution // 32
    xgrid = range(0,phase_object.resolution)
    ygrid = range(0,phase_object.resolution)
    x,y = np.meshgrid(xgrid,ygrid)

    # Determine plot name
    plot_name = "2D phase plot"
    if name != "":
        plot_name += " of " + name

    plt.title(plot_name)
    plt.imshow(magnitudes, cmap='bwr', origin='lower')
    if show_arrows:
        plt.quiver(x[::skip,::skip], y[::skip,::skip],
                plot_space_x[::skip,::skip],
                plot_space_y[::skip,::skip],
                color='r',
                scale=0.1
        )
    
    plt.axis('off')
    
    if show_fig: 
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

        length=7,
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

def plot_2d_simulation(sim_data, phase_space, show_fig=True):

    plot_2d_phase_space(phase_space, name="Simulation Data")

    scale_factor = phase_space.resolution / phase_space.dim_length

    if phase_space.four_quadrant:
        offset = phase_space.resolution // 2
    else:
        offset = 0.0

    x = sim_data[:,0]*scale_factor + offset
    y = sim_data[:,1]*scale_factor + offset
    
    plt.plot(x, y, color='#29FF22')

    if show_fig:
        plt.show()

def plot_reconstructed_space(phase_space, show_fig=True):
    """Compile and display a recontructed phase space from the 
    k-means data, and compute the RMSE.
    """

    reconstruct_space = []
    for ptr in phase_space.pointer_space.flatten():
        reconstruct_space.append(phase_space.pointer_means[ptr])
    
    space_shape = np.shape(phase_space.pointer_space) + (phase_space.dimensions,)
    reconstruct_space = np.reshape(reconstruct_space, space_shape)

    # Compute the RMSE between the representations.
    difs = reconstruct_space.flatten() - phase_space.phase_space.flatten()
    squares = difs ** 2
    rmse = sqrt(np.sum(squares) / len(difs))
    print("RMSE for the k-means representation: {:.5f}".format(rmse))

    # Compute an MSE for each vector.
    difs = reconstruct_space - phase_space.phase_space
    squares = np.square(difs)
    mse_map = np.sum(squares, axis=2)

    grad_field_compressed = np.sqrt((reconstruct_space[:,:,0]**2 + reconstruct_space[:,:,1]**2))
    grad_field_compressed = np.rot90(grad_field_compressed)

    plt.subplot(1,3,1)
    plt.title("Compressed Phase Space")
    plt.imshow(grad_field_compressed, cmap='seismic')

    grad_field = np.sqrt((phase_space.phase_space[:,:,0]**2 + phase_space.phase_space[:,:,1]**2))
    grad_field = np.rot90(grad_field)

    plt.subplot(1,3,2)
    plt.title("Original Phase Space")
    plt.imshow(grad_field, cmap='seismic')

    plt.subplot(1,3,3)
    plt.title("RMSE between representions")
    plt.imshow(mse_map, cmap='hot')

    if show_fig:
        plt.show()