import numpy as np
import json 
from bitstring import BitArray

class PhaseSpace:

    def __init__(self, ode_system, dt, resolution, max_limit, four_quadrant=True):
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

        self.compile_to_binary()
                
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

        flat_bin_strs = []

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
            
            print(bin_vals)

        return flat_bin_strs