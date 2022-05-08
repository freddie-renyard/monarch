from serial import Serial
import time 
from bitstring import BitArray

class UART:

    def __init__(self, baud=3000000, port_name="/dev/tty.usbserial-FT4ZS6I31"):
        """Initialises a serial link to a MONArch device.
        """

        self.baud = baud
        self.port_name = port_name

        # Model Parameters - will be extracted from the PhaseSpace object.
        self.dims = 2
        self.scale_factor = 1.0

        # Receive parameters
        self.bytes_per_dim = 3 # Bit depth of 24 for each dimension
        self.bytes_to_rx = self.bytes_per_dim * self.dims
        
        self.begin_serial(timeout = 5)
    
    def begin_serial(self, timeout):
        """Begin the serial communication to the FPGA.
        """

        while True:
            try:
                print("Attempting to connect to device...")
                self.serial_link = Serial(
                    port = self.port_name,
                    baudrate = self.baud
                )
                self.reader = ReadLine(self.serial_link)
                break
            except:
                print("Failed to connect to {}. Reattempting...".format(self.port_name))

            time.sleep(1)

        print("Connection to {} established!".format(self.port_name))

    def primary_eval(self, timesteps = 1):
        """Runs a given number of timesteps on the device.
        Used for evaluating the arithmetic pipeline before
        control and programming logic is added.
        """

        output_data = []
        for i in range(timesteps):
            # Write two bytes of random data to the device to trigger a timestep
            self.serial_link.write(bytes(2))

            output_data.append(self.receive_data())

        return output_data

    def receive_data(self):
        """Reads the correct number of bytes from the port for one timestep.
        """

        state_vec = [] 

        rx_bytes = self.reader.readline(self.bytes_to_rx)
        
        # Extract the value of the state vector in each dimension 
        # from the bytes read from the serial port.
        for i in range(self.dims):
            byte_value = rx_bytes[i * self.bytes_per_dim : (i+1) * self.bytes_per_dim] 
            int_val = BitArray(byte_value).int
            state_vec.append(int_val / self.scale_factor)

        return state_vec

# This code is inspired by GitHub user skoehler's code from the 
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