from re import I
import numpy as np
import json 
from bitstring import BitArray

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import math
from math import log2, ceil

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
    
        # Generate a linspace which represents each dimension in the input system.
        if four_quadrant:
            self.linspace = np.linspace(-max_limit, max_limit, num=resolution)
        else:
            self.linspace = np.linspace(0, max_limit, num=resolution)

        # Create the output phase space.
        phase_space_shape = self.space_shape + [self.dimensions]
        self.phase_space = np.zeros(phase_space_shape)
        
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
        self.k = 2 ** 8
        self.pointer_space, self.pointer_means = self.k_means_split(self.k, plot_verbose=False)

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

            # Compute an MSE for each vector.
            difs = reconstruct_space - self.phase_space
            squares = np.square(difs)
            mse_map = np.sum(squares, axis=2)

            grad_field_compressed = np.sqrt((reconstruct_space[:,:,0]**2 + reconstruct_space[:,:,1]**2))
            grad_field_compressed = np.rot90(grad_field_compressed)

            plt.subplot(1,3,1)
            plt.title("Compressed Phase Space")
            plt.imshow(grad_field_compressed, cmap='seismic')

            grad_field = np.sqrt((self.phase_space[:,:,0]**2 + self.phase_space[:,:,1]**2))
            grad_field = np.rot90(grad_field)

            plt.subplot(1,3,2)
            plt.title("Original Phase Space")
            plt.imshow(grad_field, cmap='seismic')

            plt.subplot(1,3,3)
            plt.title("RMSE between representions")
            plt.imshow(mse_map, cmap='hot')

            plt.show()

        if report_mem_usage:
            word_depth = 16 # Depth of main vector word.
            ptr_depth = math.ceil(math.log2(self.k)) # Depth of k-means pointer

            # Determine the number of bits needed to store
            # the original phase space
            bits = np.prod(np.shape(self.phase_space)) * word_depth
            print("Memory needed to store the whole phase space: {:.1f} kbit".format(bits / 1000.0))

            # Determine the number of bits needed to store the pointer-compressed space
            bits = np.prod(np.shape(self.pointer_space)) * ptr_depth + np.prod(np.shape(self.pointer_means)) * word_depth
            print("Memory needed to store the compressed phase space: {:.1f} kbit".format(bits / 1000.0))
        
        # Get the hardware parameters and set the system bit depths.
        with open("monarch/hardware_params.json") as file:
            hardware_params = json.load(file)
        
        self.datapath_size = hardware_params["datapath_size"]
        self.memory_size = hardware_params["memory_size"]

        # Compile radix parameters.
        int_dynamic_range = ceil(log2(max_limit + 1)) # Add one for the sign bit.
        self.radix = self.datapath_size - int_dynamic_range

        # Calculate real to address conversion parameters
        self.dim_width = ceil(log2(resolution))
        
        if four_quadrant:
            self.real_to_addr = self.datapath_size - self.dim_width
        else:
            self.real_to_addr = self.datapath_size - 1 - self.dim_width

        print("Radix: {}   REAL_TO_ADDR: {}   DIM_WIDTH: {}".format(self.radix, hex(self.real_to_addr), hex(self.dim_width)))

        # Compile and save the pointers to a binary.
        bin_pointers = self.compile_pointers_to_bin(self.pointer_space)
        self.save_to_file(bin_pointers, "pointers")

        # Compile and save each component dimension of the delta vectors to a binary.
        for i in range(self.dimensions):
            vec_lst = self.compile_vecs_to_bin(self.pointer_means[:, i])
            self.save_to_file(vec_lst, "vec_components_dim_{}".format(i))

    def increment_addr(self, index_lst):
        
        for i in range(len(index_lst)):
            if index_lst[i] == (self.resolution-1):
                index_lst[i] = 0
            else:
                index_lst[i] += 1
                break

        return index_lst

    def compile_pointers_to_bin(self, pointer_space):
        
        flat_ptrs = pointer_space.flatten()
 
        # Compute the required depth to store the number of vectors
        ptr_depth = math.ceil(math.log2(self.k))
    
        ptr_bins = []
        for ptr in flat_ptrs:
            binary = str(BitArray(
                    uint=ptr, 
                    length=ptr_depth
                    ).bin
                )
            ptr_bins.append(binary)
        
        return ptr_bins

    def compile_vecs_to_bin(self, vec_lst):
        """Compile a 1D list of vectors to a list of binary strings.
        """

        scale_factor = 2 ** self.radix

        vec_lst *= scale_factor
        vec_lst = vec_lst.astype(int)

        bin_vals = []
        for vector_component in vec_lst:
            binary = str(BitArray(
                int = vector_component, 
                length = self.memory_size
                ).bin
            )
            bin_vals.append(binary)

        return bin_vals

    def save_to_file(self, bin_lst, file_name):
        """Save a list of binary strings to a .mem file.
        """

        filepath = "monarch/cache/" + file_name + ".mem"

        with open(filepath, "w+") as file:
            for component in bin_lst:
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

        """Options for the pre-processing stage:
        1. k-means on raw vectors (implemented)
        2. k-means on normalised vectors - RMSE is worse than above option.
        3. k-means on log-transformed vectors - RMSE is worse than above option.
        4. k-means on separated vector components (1D), removing the directionality.

        Test when simulating. Test against existing full precision MATLAB simulations.
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