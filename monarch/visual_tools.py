from multiprocessing.sharedctypes import Value
import numpy as np
from matplotlib import pyplot as plt

def plot_2d_phase_space(phase_object, name=""):
    """Plots a 2D phase diagram for a given PhaseSpace object and shows
    the output as a quiver plot superimposed on a image plot of the total
    gradient vector magnitude.
    """

    if phase_object.dimensions != 2:
        raise ValueError("Dimensionality of the input PhaseSpace Object is not 2.")

    # Compute the magnitude of each vector.
    magnitudes = np.sqrt((phase_object.phase_space[0, :, :]**2 +  phase_object.phase_space[1, :, :]**2))

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
    plt.quiver(x[::skip,::skip], y[::skip,::skip],
            phase_object.phase_space[1, ::skip,::skip],
            -phase_object.phase_space[0, ::skip,::skip],
            color='r',
            scale=100)
    
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