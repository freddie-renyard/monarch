from bitstring import BitArray

def convert_to_fixed(target, width, radix):

    scale_factor = 2 ** radix
    scaled_val = int(target * scale_factor)
    
    bin_str = str(BitArray(int=scaled_val, length=width).bin)

    return bin_str

def convert_to_hex(number):
    if len(number) % 4 == 0:
        return str(BitArray(bin=number))[2:]
    else:
        raise Exception("MONARCH - The width of the binary data is not evenly divisble by 4 and so cannot be converted to hex")

if __name__ == "__main__":
    bin_num = convert_to_fixed(-0.75, 16, 8)
    print(convert_to_hex(bin_num))