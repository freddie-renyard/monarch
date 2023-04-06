import os
import json
import serial
import time
from math import ceil
from bitstring import BitArray
from parsers.float_compiler import BinCompiler
from matplotlib import pyplot as plt
import numpy as np

class FPGAPort:

    def __init__(self):
        
        self.port_name = "/dev/cu.usbserial-A904DPPI"
        self.timeout = 10
        self.baud = 115200

        # Open architecture database.
        script_dir = os.path.dirname(__file__)
        rel_path = "../parsers/arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            self.n_man = dbs['sys_params']['datapath_mantissa']
            self.n_exp = dbs['sys_params']['datapath_exponent']

        # Acquire a connection to the FPGA
        self.open_uart()

        sim_dims = 2
        timesteps = 500

        out_data = None
        for i in range(10):
            self.run_timesteps(timesteps)
            data = self.receive_data(
                sim_dims, 
                timesteps,
                1 + self.n_exp + self.n_man
            )

            if i:
                out_data = np.concatenate((out_data, data))
            else:
                out_data = data
            
            print(np.shape(out_data))

        plt.plot(out_data)
        plt.show()
        
    def open_uart(self):
        # Open the serial port 
        
         # Define the number of connection attempts
        rest_interval = 1 
        max_attempts = self.timeout // rest_interval
        attempts = 0
        
        while True: 
            try:
                self.port = serial.Serial(self.port_name, baudrate=self.baud)
                self.reader = ReadLine(self.port)
                print('MONArch: Opened serial port to device at', self.port_name)
                break
            except:
                print('MONArch: Failed to connect to {}. Reattempting...'.format(self.port_name))
                attempts += 1
                time.sleep(rest_interval)

            if attempts >= max_attempts and max_attempts != 0:
                print("MONArch: ERROR: Serial connection to {} failed.".format(self.port_name))
                break
    
    def run_timesteps(self, timesteps):
        # Sends the appropriate codes to the FPGA to run the specified number of simulation timesteps

        reset_code = bytes([0x00])
        self.port.write(bytes([0x00, 0x00]) + reset_code)

        dt_code = bytes([0x01])
        bytes_dts = timesteps.to_bytes(2, "big")
        self.port.write(bytes_dts + dt_code)

    def receive_data(self, sim_dims, timesteps, n_data):
        
        # Compute the number of bytes per data word
        bytes_per_word = ceil(n_data / 8)
        bytes_per_packet = sim_dims * timesteps * bytes_per_word
        bytes_lst = self.reader.readline(bytes_per_packet)

        # Convert byte list to float list
        grouped_bytes = [bytes_lst[x:x+bytes_per_word] for x in range(0, bytes_per_packet, bytes_per_word)]
        grouped_binstr = [BitArray(bytes=x).bin for x in grouped_bytes]
        debug = [BitArray(bytes=x).hex for x in grouped_bytes]

        grouped_floats = [BinCompiler.decode_custom_float(x, self.n_man, self.n_exp) for x in grouped_binstr]
        final_dat = [grouped_floats[x:x+sim_dims] for x in range(0, sim_dims * timesteps, sim_dims)]
        
        return final_dat
    
# This code is inspired by GitHub user skoehler's code from the 
# following pyserial issue: https://github.com/pyserial/pyserial/issues/216
class ReadLine:
    def __init__(self, s):
        self.s = s
    
    def readline(self, bytes_to_read):
        
        buf = bytearray()
        buf_size = 0
        while buf_size < bytes_to_read:
            i = max(1, min(2048, self.s.in_waiting))
            buf += self.s.read(i)
            buf_size += i
        return buf