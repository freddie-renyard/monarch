import numpy as np
import json 
from bitstring import BitArray

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import math

class PhaseSpace:

    def __init__(self, ode_system, dt, resolution, max_limit, four_quadrant=True, verbose=False, report_mem_usage=True):
        """ A class which contains the compiled phase space data, along with methods 
        for compiling the equations passed to this class on initialisation.
        """

        # A tuple of lambda functions which describe the system to be modelled.
        self.ode_system = ode_system
        self.dimensions = len(ode_system)

        self.resolution = resolution
        self.space_shape = [resolution] * self.dimensions
    
        # Generate a linspace which represents each dimesion in the input system.
        if four_quadrant:
            self.linspace = np.linspace(-max_limit, max_limit, num=resolution)
        else:
            self.linspace = np.linspace(0, max_limit, num=resolution)

        # Create the output phase space.
        phase_space_shape = self.space_shape + [self.dimensions]
        self.phase_space = np.zeros(phase_space_shape)
        
        # Create an array of zeroes to hold the addresses when compiling.

        for dim_i, delta_eqn in enumerate(self.ode_system):
            # Extract each equation and compile the deltas for that direction
            # into a tensor.
            break_next = False
            ind_arr = [0] * self.dimensions
            while True:
                
                # Get the current entry in phase space.
                address = tuple(ind_arr + [dim_i])
                
                # Get the real values for the current address from the linspace.
                dim_args = []
                for i in ind_arr:
                    dim_args.append(self.linspace[i])
                
                self.phase_space[address] = delta_eqn(*dim_args)
                
                ind_arr = self.increment_addr(ind_arr)

                if break_next:
                    break

                # Set a break next condition to ensure all the addresses are
                # calculated.
                if sum(ind_arr) == (self.resolution-1)*self.dimensions:
                    break_next = True
        
        # Mutliply all the values in the phase space by the timestep
        # to make evaluation of the ODE with Euler's method faster in hardware.
        self.phase_space *= dt

        # Compile the k-means pointer space and the associated means.
        k = 2 ** 5
        self.pointer_space, self.pointer_means = self.k_means_split(k, plot_verbose=False)

        # Compile and display a recontructed phase space from the 
        # k-means data, and compute the RMSE.
        if verbose:
            reconstruct_space = []
            for ptr in self.pointer_space.flatten():
                reconstruct_space.append(self.pointer_means[ptr])
            
            space_shape = np.shape(self.pointer_space) + (self.dimensions,)
            reconstruct_space = np.reshape(reconstruct_space, space_shape)
        
            # Compute the RMSE between the representations.
            difs = reconstruct_space.flatten() - self.phase_space.flatten()
            squares = difs ** 2
            rmse = math.sqrt(np.sum(squares) / len(difs))
            print("RMSE for the k-means representation: {:.5f}".format(rmse))

            plt.subplot(1,2,1)
            plt.imshow(reconstruct_space[:,:,1])
            plt.subplot(1,2,2)
            plt.imshow(self.phase_space[:,:,1])

            plt.show()

        # Compile to binary
        self.bin_space = self.compile_to_binary()
        
        self.save_to_file()
                
    def increment_addr(self, index_lst):
        
        for i in range(len(index_lst)):
            if index_lst[i] == (self.resolution-1):
                index_lst[i] = 0
            else:
                index_lst[i] += 1
                break

        return index_lst

    def compile_to_binary(self):
        """Compile the phase space to a binary representation.
        """

        flat_bin_lsts = []

        # Get the compiler parameters
        with open("monarch/hardware_params.json") as file:
            comp_params = json.load(file)

        scale_factor = 2 ** comp_params["delta_radix"]

        # Extract the vector components for each dimension of phase space.
        for vector_comps in np.moveaxis(self.phase_space, -1, 0):

            flat_vectors = vector_comps.flatten()
            
            flat_vectors *= scale_factor
            flat_vectors = flat_vectors.astype(int)

            bin_vals = []
            for vector_component in flat_vectors:
                binary = str(BitArray(
                    int=vector_component, 
                    length=comp_params["delta_depth"]
                    ).bin
                )
                bin_vals.append(binary)

            flat_bin_lsts.append(bin_vals)
            
        return flat_bin_lsts

    def save_to_file(self):
        """ Save a compiled memory to a file in the temporary cache.
        """

        filepath = "monarch/cache/phase_space_dim_{}.mem"

        for dim_i, binary_lst in enumerate(self.bin_space):
            with open(filepath.format(dim_i), "w+") as file:
                for component in binary_lst:
                    file.write((component + " \n"))

    def k_means_split(self, k, plot_verbose=False):
        """Split a phase space of arbitrary dimesions into K vectors.
        Return the vectors and an associated pointer phase space as
        numpy arrays.
        """

        # Flatten the phase space so that the vectors can be visualised
        # outside of an image representation.
        space_shape = np.shape(self.phase_space)
        new_shape = (np.prod(space_shape[:-1]), space_shape[-1])
        vectors = np.reshape(self.phase_space, new_shape)

        """Options for this stage:
        1. k-means on raw vectors (implemented)
        2. k-means on normalised vectors - tested, worse than above option.
        3. k-means on log-transformed vectors - tested, worse than above option.
        4. k-means on separated vector components (1D), removing the directionality.
        """

        kmeans = KMeans(k)
        kmeans.fit(vectors)

        id_clusters = kmeans.fit_predict(vectors)

        # Compute the un-normalised centroid vector for each 
        # k-means class.
        means = np.zeros((k, self.dimensions))
        counts = np.zeros((k))
        
        for i, addr in enumerate(id_clusters):
            counts[addr] += 1
            means[addr, :] = means[addr, :] + vectors[i, :]

        # Divide the means by the number of cells containing a state vector belonging
        # to it's k-means class. TODO double-check validity of these vectors.
        means = means / np.repeat(np.expand_dims(counts, axis=1), self.dimensions, axis=1)

        # Reshape the identifiers into the orginal tensor shape.
        id_struct = np.reshape(id_clusters, space_shape[:-1])

        if plot_verbose and self.dimensions == 2:

            plt.title("")
            plt.scatter(vectors[:, 0], vectors[:, 1], c = id_clusters, cmap='rainbow')

            plt.subplot(1,3,1)
            plt.title("K-means Clustering \nNormalised Phase Space")
            plt.imshow(id_struct, cmap='gist_rainbow')

            plt.subplot(1,3,2)
            plt.title("Vector components for X")
            plt.imshow(self.phase_space[:,:,0])

            plt.subplot(1,3,3)
            plt.title("Vector components for Y")
            plt.imshow(self.phase_space[:,:,1])

            plt.show()

        return id_struct, means