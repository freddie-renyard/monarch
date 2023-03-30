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
    lut_res = target_dat["table_size"]
    shift_val = n_input - ceil(log2(target_dat["table_size"]))

    # Determine all the possible input binary numbers
    in_bin_str = [BinCompiler.compile_to_uint(x << shift_val, n_input, 0) for x in range(0, lut_res)]

    # Determine what value this binary actually represents under 
    # floating point interpretation.
    in_float_vals = [BinCompiler.decode_custom_float(x, n_mantissa_in, n_exp_in) for x in in_bin_str]

    # Compute the exponential value for each value.
    out_exp_vals = [target_fn(x) for x in in_float_vals]
    out_bin = [BinCompiler.compile_to_float(x, n_mantissa_out, n_exp_out) for x in out_exp_vals] 

    save_lut_file(out_bin, False, arch_fn, target_dat, shift_val, width, 0, 0)

def compile_custom_exp_lut(n_man, n_exp):
    # Compiles an optimised lookup table for the exponential function.

    target_dat, width, _ = get_arch_dat("e")
    table_size = target_dat["table_size"]
    
    # Determine the minimum and maximum value that can be represented by
    # the floating point format.
    max_val = BinCompiler.decode_custom_float(
        "0{}".format("1" * (n_man + n_exp)),
        n_man,
        n_exp
    )

    min_val = BinCompiler.decode_custom_float(
        "0{}{}".format("0" * (n_exp-1) + "1", "0" * (n_man)),
        n_man,
        n_exp
    )

    # Compile a full float for every possible mantissa value from '0 to '1
    man_bins = [BinCompiler.compile_to_uint(x, n_man+1, 0)[2:] for x in range(2 ** n_man)]
    
    # Compute the value of the exponential function for each mantissa value, as if there is an
    man_vals = [BinCompiler.decode_custom_float(
        "0" + ("1" + "0" * (n_exp-1)) + x,
        n_man,
        n_exp
    ) for x in man_bins]

    # Get the value of the exponential function at every mantissa value
    man_exp_vals = [exp(x) for x in man_vals]

    # Decimate the table to bring the mantissa into representation and determine
    # the shift value
    n_table = log2(table_size)

    if not n_table.is_integer():
        raise Exception("MONARCH - LUT table size must be a power of two; is currently {}".format(table_size))
    
    n_table = int(n_table)
    shift_size = n_man - n_table
    
    if shift_size < 0:
        raise Exception("The mantissa depth ({}) must be larger or equal to table depth ({})".format(n_man, n_table))

    window = 2 ** (shift_size)
    
    reduc_exp_vals = [np.mean(man_exp_vals[x*window:(x+1)*window])for x in range(table_size)]
    
    # Compile the decimated exponent output values to floating point
    exp_bins_full = [BinCompiler.compile_to_float(x, n_man, n_exp) for x in reduc_exp_vals]
    
    # Strip off the exponent and sign, and append to correction number to determine final values.
    man_bins_final = []
    corr_size = 2
    for bin in exp_bins_full:

        # Compute the exponent correction needed to normalise the value
        corr = int(bin[1:n_exp+1], 2) - (2 ** (n_exp-1))
        corr_bin = BinCompiler.compile_to_uint(corr, corr_size, 0)
        
        # Strip off exponent and sign and concatenate bits.
        man_bins_final.append(
            corr_bin + bin[n_exp+1:]
        )

    # Determine every possible exponent binary, inclusive of zero.
    exp_bins = [BinCompiler.compile_to_uint(x, n_exp, 0) for x in range(2 ** n_exp)]

    # Concatenate the signs onto the exponents 
    exp_and_sign_bins = [*["0" + bin for bin in exp_bins], *["1" + bin for bin in exp_bins]]

    # For each exponent, determine if the e^x output values are within it's limits.
    exp_bins_final = []
    for exp_bin in exp_and_sign_bins:
        min_exp_val = BinCompiler.decode_custom_float(
            exp_bin + "0" * n_man, n_man, n_exp
        )
        max_exp_val = BinCompiler.decode_custom_float(
            exp_bin + "1" * n_man, n_man, n_exp
        )
 
        try:
            min_e_val = exp(min_exp_val)
        except:
            # Make the output value the maximum value that the datatype can represent
            # if the function overflows.
            min_e_val = max_val
        
        # Compute the exponent map, starting at the minumum value.
        compiled_exp = BinCompiler.compile_to_float(min_e_val, n_man, n_exp)[:n_exp+1]

        exp_bins_final.append(compiled_exp)
    
    save_float_lut_file(
        "e",
        man_bins_final,
        exp_bins_final,
        len(man_bins_final),
        len(exp_bins_final),
        corr_size
    )

def save_float_lut_file(arch_fn, man_table, exp_table, man_tab_size, exp_tab_size, corr_size):

    file_name = "monarch/cache/{}_lut_float_{}.mem"
    with open(file_name.format(arch_fn, "man"), "w+") as file:

        file.write("// Mantissa lookup table for {} function, {} entries. \n".format(arch_fn, man_tab_size))
        for val in man_table:
            file.write(str(val) + "\n")
    
    with open(file_name.format(arch_fn, "exp"), "w+") as file:
        file.write("// Exponent and sign lookup table for {} function, {} entries. \n".format(arch_fn, exp_tab_size))
        for val in exp_table:
            file.write(str(val) + "\n")

    # Write all parameters needed for this module to function properly to a pkg file.
    with open("monarch/dbs/lut_float_pkg_template.sv") as file:
        pkg_str = file.read()
        pkg_str = pkg_str.replace("<man_table_size>", str(man_tab_size))
        pkg_str = pkg_str.replace("<exp_table_size>", str(exp_tab_size))
        pkg_str = pkg_str.replace("<corr_size>", str(corr_size))
    
    with open("monarch/cache/lut_pkg.sv", "w+") as file:
        file.write(pkg_str)

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
