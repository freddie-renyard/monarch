import numpy as np
import os 
import json
from math import ceil, log2, floor
from matplotlib import pyplot as plt
from parsers.bin_compiler import convert_to_fixed, twos_to_uint, bin_to_int
from parsers.float_compiler import BinCompiler
from math import exp

def get_max_val (width, radix):
    if width == 0: return 0.0
    return int("0b0" + "1"*(width-1), 2) / (2.0 ** radix)

def get_min_abs_val(width, radix):
    if width == 0: return 0.0
    return int("0b" + "0"*(width-1) + "1", 2) / (2.0 ** radix)

def get_arch_dat(arch_fn):

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

    return target_dat, width, radix

def generate_lut(arch_fn):
    # Generates look up tables for the LUT unit in hardware.

    target_dat, width, radix = get_arch_dat(arch_fn)

    target_fn = eval(target_dat["fn"])

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
        
        float_ins = []
        bin_ins  = []
        bin_outs = []
        sign_offset = int(target_dat["table_size"] / 2)

        for i in range(target_dat["table_size"]):
            #inputs = sample_in_space[int(i*sample_rat):int((i+1)*sample_rat)]
            #outputs = vfunc(inputs)
            in_val = (i - sign_offset) << shift_val

            in_val = float(in_val) / float(2 ** radix)

            float_ins.append(in_val)

            bin_in = convert_to_fixed(in_val, width, radix)
            bin_out = convert_to_fixed(target_fn(in_val), width, radix)
            bin_ins.append(bin_in)
            bin_outs.append(bin_out)

        # Reorder the address space to allow for contiguous memory representation.
        int_bin_outs = [bin_to_int("0" + x[width - targ_range - shift_val:width - targ_range]) for x in bin_ins]

        sorted_is = np.argsort(int_bin_outs)
        sorted_table = np.array(bin_outs)[sorted_is]

        save_lut_file(sorted_table, True, arch_fn, target_dat, shift_val, width, max_bin, min_bin)
    else:
        raise Exception("MONARCH - LUT instruction target function not recognised: {}".format(arch_fn))

def compile_float_lut(arch_fn, n_mantissa_in, n_exp_in, n_mantissa_out, n_exp_out):
    """ Compiles an lookup table for a target_function function with
    floating point inputs and floating point outputs.
    """

    target_dat, width, _ = get_arch_dat(arch_fn)
    target_fn = eval(target_dat["fn"])
    
    n_input = 1 + n_mantissa_in + n_exp_in
    lut_res = 2 ** n_input

    # Determine all the possible input binary numbers
    in_bin_str = [BinCompiler.compile_to_uint(x, n_input, 0) for x in range(0, lut_res)]

    # Determine what value this binary actually represents under 
    # floating point interpretation.
    in_float_vals = [BinCompiler.decode_custom_float(x, n_mantissa_in, n_exp_in) for x in in_bin_str]

    # Compute the exponential value for each value.
    out_exp_vals = [target_fn(x) for x in in_float_vals]
    out_bin = [BinCompiler.compile_to_float(x, n_mantissa_out, n_exp_out) for x in out_exp_vals] 

    save_lut_file(out_bin, False, arch_fn, target_dat, 0, width, 0, 0)

def save_lut_file(table, fixed_point, arch_fn, target_dat, shift_val=0, width=1, max_bin=0, min_bin=0):
    # Saves the LUT for a given function.

    # Write table to file.
    if fixed_point: 
        float_str = "" 
    else: 
        float_str = "_float"

    file_name = "monarch/cache/{}_lut{}.mem"
    with open(file_name.format(arch_fn, float_str), "w+") as file:
        file.write("// Lookup table for {} function, {} entries. \n".format(arch_fn, target_dat["table_size"]))
        for val in table:
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
