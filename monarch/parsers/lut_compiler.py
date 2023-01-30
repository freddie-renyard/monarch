from ast import excepthandler
from doctest import testfile
import numpy as np
import os 
import json
from math import ceil, log2
from matplotlib import pyplot as plt

def get_max_val(width, radix):
    if width == 0: return 0.0
    return int("0b" + "1"*(width), 2) / (2.0 ** radix)

def generate_lut(addr_depth, arch_fn, width, radix):
    # Generates look up tables for the LUT unit in hardware.

    # Open architecture database.
    script_dir = os.path.dirname(__file__)
    rel_path = "arch_dbs.json"
    abs_file_path = os.path.join(script_dir, rel_path)
    
    with open(abs_file_path) as file:
        dbs = json.loads(file.read())

    try:
        target_dat = dbs["lut_functions"][arch_fn]
    except:
        raise Exception("MONARCH - Operation not recognised: {}".format(arch_fn))

    target_fn = eval(target_dat["fn"])
    vfunc = np.vectorize(target_fn)

    test_x = []
    test_y = []

    for sign in range(2):
        
        entry_fn = eval(target_dat["entries_fn"])
            
        uint_width = width - 1
        range_pairs = [[get_max_val(x, radix), get_max_val(x+1, radix)] for x in range(uint_width)]
        range_pairs = np.array(range_pairs)
        range_pairs_ld_zs = [x for x in range(0, uint_width)][::-1]

        # Sign the range pairs if needed.
        if sign: 
            range_pairs *= -1.0

        for (min_in, max_in), ld_zs in zip(range_pairs, range_pairs_ld_zs):
            
            # Normalise the number of leading zeros to within [0, 1] <-> [min, max]
            norm_ld_zs = float(ld_zs) / (uint_width-1)

            # Compute the size of the LUT for the specific range group.
            entry_n = 2 ** ceil(log2(entry_fn(norm_ld_zs)))
            
            # Generate the linspace needed to compute the target function.
            out_space = np.linspace(min_in, max_in, entry_n)
            out_vals = vfunc(out_space)

            test_x = [*test_x, *out_space]
            test_y = [*test_y, *out_vals]

    # Generate test data
    linspace = np.linspace((-2.0 ** uint_width) / (2.0 ** radix), (2.0 ** uint_width) / (2.0 ** radix))
    y_dat = vfunc(linspace)

    plt.plot(linspace, y_dat, color='red')
    plt.scatter(test_x, test_y)
    plt.show()

if __name__ == "__main__":
    generate_lut(8, "e", 16, 12)
