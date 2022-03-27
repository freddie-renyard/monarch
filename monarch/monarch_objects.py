from operator import index
from re import I
from unittest import result
import numpy as np

from matplotlib import pyplot as plt

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
                
    def increment_addr(self, index_lst):
        
        for i in range(len(index_lst)):
            if index_lst[i] == (self.resolution-1):
                index_lst[i] = 0
            else:
                index_lst[i] += 1
                break

        return index_lst