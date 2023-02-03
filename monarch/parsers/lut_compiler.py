from ast import excepthandler
from doctest import testfile
import numpy as np
import os 
import json
from math import ceil, log2, floor
from matplotlib import pyplot as plt
from parsers.bin_compiler import convert_to_fixed, twos_to_uint, bin_to_int

def get_max_val (width, radix):
    if width == 0: return 0.0
    return int("0b0" + "1"*(width-1), 2) / (2.0 ** radix)

def get_min_abs_val(width, radix):
    if width == 0: return 0.0
    return int("0b" + "0"*(width-1) + "1", 2) / (2.0 ** radix)

def generate_lut(arch_fn):
    # Generates look up tables for the LUT unit in hardware.

    # Open architecture database.
    script_dir = os.path.dirname(__file__)
    rel_path = "arch_dbs.json"
    abs_file_path = os.path.join(script_dir, rel_path)
    
    with open(abs_file_path) as file:
        dbs = json.loads(file.read())

    width = dbs["sys_params"]["datapath_width"]
    radix = dbs["sys_params"]["datapath_radix"]
    try:
        target_dat = dbs["lut_functions"][arch_fn]
    except:
        raise Exception("MONARCH - Operation not recognised: {}".format(arch_fn))

    target_fn = eval(target_dat["fn"])
    vfunc = np.vectorize(target_fn)

    if arch_fn == "e":
        # Calculate the maximum input value that it is worth synthesising
        # the look-up table to.
        max_out_val = get_max_val(width, radix)
        max_in_val = np.log(max_out_val)
        
        min_out_val = get_min_abs_val(width, radix)
        min_in_val = np.log(min_out_val)
        
        # Clip the values to the nearest lower power of two to ensure full
        # use of the table size.
        max_in_val = float(2 ** floor(log2(abs(max_in_val)))) 
        min_in_val = -float(2 ** floor(log2(abs(min_in_val))))

        max_bin = convert_to_fixed(max_in_val, width, radix)
        min_bin = convert_to_fixed(min_in_val, width, radix)

        max_int = bin_to_int(max_bin)
        min_int = -bin_to_int(twos_to_uint(min_bin, width))
        val_range = max_int - min_int
        
        clog_range = ceil(log2(val_range))
        targ_range = ceil(log2(target_dat["table_size"]))
        shift_val = clog_range - targ_range
        
        bin_ins  = []
        bin_outs = []
        sign_offset = int(target_dat["table_size"] / 2)

        for i in range(target_dat["table_size"]):
            #inputs = sample_in_space[int(i*sample_rat):int((i+1)*sample_rat)]
            #outputs = vfunc(inputs)
            in_val = (i - sign_offset) << shift_val
            in_val = float(in_val) / float(2 ** radix)

            bin_in = convert_to_fixed(in_val, width, radix)
            bin_out = convert_to_fixed(target_fn(in_val), width, radix)
            bin_ins.append(bin_in)
            bin_outs.append(bin_out)

        # Reorder the address space to allow for contiguous memory representation.
        int_bin_outs = [bin_to_int("0" + x[width - targ_range - shift_val:width - targ_range]) for x in bin_ins]

        sorted_is = np.argsort(int_bin_outs)
        sorted_table = np.array(bin_outs)[sorted_is]

        # Write table to file.
        with open("monarch/cache/{}_lut.mem".format(arch_fn), "w+") as file:
            file.write("// Lookup table for {} function, {} entries. \n".format(arch_fn, target_dat["table_size"]))
            for val in sorted_table:
                file.write(str(val) + "\n")

        # Write all parameters needed for this module to function properly to a pkg file.
        with open("monarch/dbs/lut_pkg_template.sv") as file:
            pkg_str = file.read()
            pkg_str = pkg_str.replace("<table_size>", str(target_dat["table_size"]))
            pkg_str = pkg_str.replace("<shift_val>", str(shift_val))
            pkg_str = pkg_str.replace("<max_val>", "{}'b".format(width) + str(max_bin))
            pkg_str = pkg_str.replace("<min_val>", "{}'b".format(width) + str(min_bin))
        
        with open("monarch/cache/lut_pkg.sv", "w+") as file:
            file.write(pkg_str)
    else:
        raise Exception("MONARCH - LUT instruction target function not recognised: {}".format(arch_fn))
