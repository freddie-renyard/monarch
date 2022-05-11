from ctypes import pointer
from re import I
import numpy as np
import json 
from bitstring import BitArray, Bits
from sklearn.cluster import KMeans
import math
from math import log2, ceil

class PhaseSpace:

    def __init__(self, ode_system, dt, resolution, max_limit, four_quadrant=True, report_mem_usage=True, compress_space=True):
        """ A class which contains the compiled phase space data, along with methods 
        for compiling the equations passed to this class on initialisation.
        """

        # A tuple of lambda functions which describe the system to be modelled.
        self.ode_system = ode_system
        self.dimensions = len(ode_system)
        self.four_quadrant = four_quadrant

        # Ensure that the max_limit is a power of 2.
        max_limit = 2 ** ceil(log2(max_limit))
        
        if four_quadrant:
            self.dim_length = max_limit * 2
        else:
            self.dim_length = max_limit

        # Get the hardware parameters.
        with open("monarch/hardware_params.json") as file:
            hardware_params = json.load(file)

        if resolution > hardware_params["max_resolution"]:
            raise(ValueError("MONArch: Resolution is above the current hardware maximum."))
        else:
            # Ensure that the resolution is a power of 2.
            self.resolution = 2 ** ceil(log2(resolution))

        self.dt = dt

        self.create_cellular_phase_space(max_limit = max_limit)

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

        # Compile the k-means pointer space and the associated means.
        self.k = 2 ** hardware_params["pointer_size"]

        if compress_space:
            self.pointer_space, self.pointer_means = self.k_means_split(self.k)

            # Compile and save the pointers to a binary.
            if four_quadrant:
                comp_ptrs = self.compile_four_quad_pointers(self.pointer_space)
            else:   
                comp_ptrs = self.pointer_space
        
            bin_pointers = self.compile_pointers_to_bin(comp_ptrs)

            self.save_to_file(bin_pointers, "pointers")

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
        
            # Compile and save each component dimension of the delta vectors to a binary.
            for i in range(self.dimensions):
                vec_lst = self.compile_vecs_to_bin(self.pointer_means[:, i])
                self.save_to_file(vec_lst, "vec_components_dim_{}".format(i))

        else:
            # Compile the uncompressed vectors.
            for i in range(self.dimensions):
                if four_quadrant:
                    comp_vecs = self.compile_four_quad_pointers(self.phase_space[..., i])
                else:   
                    comp_vecs = self.phase_space[..., i].flatten()
                
                bin_vecs = self.compile_vecs_to_bin(comp_vecs)
                self.save_to_file(bin_vecs, "vec_components_dim_{}".format(i))



    def increment_addr(self, index_lst):
        
        for i in range(len(index_lst)):
            if index_lst[i] == (self.resolution-1):
                index_lst[i] = 0
            else:
                index_lst[i] += 1
                break

        return index_lst

    def create_cellular_phase_space(self, max_limit):
        """Create the cellular phase space.
        """

        # Generate a linspace which represents each dimension in the input system.
        if self.four_quadrant:
            linspace = np.linspace(-max_limit, max_limit, num=self.resolution)
        else:
            linspace = np.linspace(0, max_limit, num=self.resolution)

        space_shape = [self.resolution] * self.dimensions
        phase_space_shape = space_shape + [self.dimensions]
        self.phase_space = np.zeros(phase_space_shape)
        
        for dim_i, delta_eqn in enumerate(self.ode_system):
            # Extract each equation and compile the deltas for that direction
            # into a tensor.
            break_next = False
            ind_arr = [0] * self.dimensions
            while True:
                
                # Get the current entry in phase space.
                address = tuple(ind_arr[::-1] + [dim_i])
                
                # Get the real values for the current address from the linspace.
                dim_args = []
                for i in ind_arr:
                    dim_args.append(linspace[i])
                
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
        self.phase_space *= self.dt

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

    def compile_four_quad_pointers(self, pointer_space):
        """Save the pointer space to a binary, but reorder the input address 
        space first to ensure that two's complement addresses are valid.
        """

        # Get linspace for the input addresses across each dimension
        linspace = np.array([i for i in range(0, self.resolution, 1)])

        # Offset the linspace to show the actual addresses 
        linspace = linspace - self.resolution // 2
        
        # Convert the linspace to 2s complement bitstrings interpreted as
        # unsigned addresses (as this is the hardware's interpretation)
        bin_linspace = []
        for addr in linspace:
            bin_linspace.append(Bits(int=addr, length=self.dim_width).uint)
        
        # Recursively reorder the pointer space.
        # TODO Test this for dimensions higher than 2.
        def reorder_elements(flat_array, linspace, dimension, max_dim):
            
            # Get the step of the array.
            step = self.resolution ** dimension
            outer_step = self.resolution ** (dimension+1)

            reord_arr = np.zeros(np.shape(flat_array))

            offset = 0
            while offset != np.shape(flat_array)[0]:

                for i, addr in enumerate(linspace):
                    window = flat_array[offset + (i*step) : offset + (i*step) + step]
                    reord_arr[offset + (addr*step) : offset + (addr*step) + step] = window 

                offset += outer_step
            
            if dimension == max_dim-1:
                return reord_arr
            else:
                return reorder_elements(reord_arr, bin_linspace, dimension+1, max_dim)

        return reorder_elements(pointer_space.flatten(), bin_linspace, 0, max_dim=self.dimensions)

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
                ).hex
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
    
    def k_means_split(self, k):
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

        return id_struct, means

    def run_simulation(self, init_state, timesteps):
        """Run a simulation of the system in Python.
        Currently uses Euler's method.
        """
        
        sim_shape = [timesteps, self.dimensions]
        sim_data = np.zeros(sim_shape)

        sim_data[0] = init_state

        for i in range(1, timesteps):
            for d in range(self.dimensions):
                sim_data[i][d] = sim_data[i-1][d] + self.dt * self.ode_system[d](*sim_data[i-1])
            if i % 400:
                print("Python Timestep: {}".format(i))
        return sim_data