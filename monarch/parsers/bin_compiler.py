from bitstring import BitArray
import numpy as np

def convert_to_fixed(target, width, radix, signed=True):

    scale_factor = 2 ** radix
    scaled_val = int(target * scale_factor)
    
    if signed:
        bin_str = str(BitArray(int=scaled_val, length=width).bin)
    else:
        bin_str = str(BitArray(uint=scaled_val, length=width).bin)

    return bin_str

def convert_to_uint(target, width):
    
    bin_str = str(BitArray(uint=target, length=width).bin)
    return bin_str

def convert_to_hex(number):
    if len(number) % 4 == 0:
        return str(BitArray(bin=number))[2:]
    else:
        raise Exception("MONARCH - The width of the binary data is not evenly divisble by 4 and so cannot be converted to hex")