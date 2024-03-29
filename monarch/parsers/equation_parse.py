from distutils.command.build import build
from email.mime import base
from parsers.cfg_compiler import extract_source_nodes
import math
import sympy
from sympy import core, S, Symbol, Function, Lambda, powsimp
import json
import os

def convert_to_str(tree):
    # Convert dictionary nodes to strings for printing with JSON dumps.
    
    if type(tree) != dict:
        return str(tree)

    converted_ins = []
    for input_node in tree['inputs']:
       converted_ins.append(convert_to_str(input_node))
    
    return {
        "op": tree['op'],
        "inputs": converted_ins
    }

def extract_equality_expr(eq):
    # Extracts two sides of an equation from a string.

    parts = eq.strip().split("\n")
    subparts = [x.split("=") for x in parts]
    subparts = [x for x in subparts if len(x) > 1]
    for i, subpart in enumerate(subparts):
        subparts[i] = [x.strip() for x in subpart]

    return subparts

def get_operation_name(op):
    if type(op) == str:
        return op
    else:
        return op.__name__

def get_operation(eq, node_index=0):

    # get type of operation at current graph depth.
    op_type = type(eq)

    atoms = eq.atoms()
    inputs = []
    for subexpr in eq.args:
        if subexpr in atoms:
            # The input is a variable, constant or number.
            inputs.append(subexpr)
        else:
            inputs.append(get_operation(subexpr))

    return {
        "op": op_type, 
        "inputs": inputs
    }

def verify_differential(eq):
    # Checks that the LHS of the equation given is in the appropriate format
    # and returns the symbol that has been differentiated.
    
    # TODO add error checking for incorrect input formats.
    symb_eq = sympy.sympify(eq)
    
    d_var_symb = symb_eq.args[0]
    var_str = str(d_var_symb)[1:]

    return sympy.Symbol(var_str)

def check_negative_var(cfg):

    if type(cfg) != dict:
        # We've reached a terminal node aka an atomic.
        return None

    if cfg['op'].__name__ == "Mul":
        if len(cfg['inputs']) >= 2:
            types = [type(x) for x in cfg['inputs']]
            if type(S.NegativeOne) in types:
                if len(cfg['inputs']) == 2:
                    # Dissolve the multiplication operation completely.
                    return [x for x in cfg['inputs'] if type(x) != type(S.NegativeOne)][0]
                else:
                    return {
                        "op": "mult",
                        "inputs": [x for x in cfg['inputs'] if type(x) != type(S.NegativeOne)]
                    }

def check_divided_var(cfg):

    if type(cfg) != dict:
        # We've reached a terminal node aka an atomic.
        return None
    
    if get_operation_name(cfg['op']) == "Pow":
        if len(cfg['inputs']) >= 2:
            types = [type(x) for x in cfg['inputs']]
            if type(S.NegativeOne) in types:
                if len(cfg['inputs']) == 2:
                    # Dissolve the power operation completely by returning the denominator variable
                    return [x for x in cfg['inputs'] if type(x) != type(S.NegativeOne)][0]

def build_adder_tree(input_ops):
    # Null operator for this particular operation (0)
    identity = Symbol('0')

    if len(input_ops) == 1:
        return input_ops[0]

    if len(input_ops) % 2 != 0:
        input_ops.append(identity)

    adder_pairs = [input_ops[i:i+2] for i in range(0, len(input_ops), 2)]
    
    ops = []
    for pair in adder_pairs:
       ops.append({
           "op": "add",
           "inputs": pair
       }) 

    return build_adder_tree(ops)

def construct_binary_adder_tree(pos_args, neg_args):
    # Recursively designs a subtractor/adder tree for the given operands.
    
    # TODO integrate the superior algorithm tree construction algorithm 
    # below into this part of the code.
    if len(pos_args) > 0:
        pos_tree = build_adder_tree(pos_args)
    else:
        pos_tree = Symbol('0')

    if len(neg_args) > 0:
        neg_tree = build_adder_tree(neg_args)
    else:
        return pos_tree

    # Subtract the positive tree from the negative one to build
    # the final tree.
    return {
        "op": "sub",
        "inputs": [pos_tree, neg_tree]
    }

def construct_balanced_tree(input_args, opcode="", add_squarers=False):
    # Constructs a tree for a given operation which minimises edges 
    # and maximum depth of the graph.

    if len(input_args) % 2 != 0:
        return {
            "op": opcode,
            "inputs": [
                input_args[0],
                construct_balanced_tree(input_args[1:], opcode, add_squarers=add_squarers)
            ]
        }
    elif len(input_args) == 2:
        if add_squarers:
            return {
                "op": 'square',
                "inputs": [input_args[0]]
            }
        else:
            return {
                "op": opcode,
                "inputs": input_args
            }
    else:
        return {
            "op": opcode,
            "inputs": [
                construct_balanced_tree(input_args[0:len(input_args)//2], opcode, add_squarers=add_squarers),
                construct_balanced_tree(input_args[len(input_args)//2:], opcode, add_squarers=add_squarers),
            ]
        }

def verify_base(base):
    # Verify that a base of an exponentiation is supported by MONArch.
    
    # Open architecture database.
    script_dir = os.path.dirname(__file__)
    rel_path = "arch_dbs.json"
    abs_file_path = os.path.join(script_dir, rel_path)
    
    with open(abs_file_path) as file:
        dbs = json.loads(file.read())

    return str(base) in dbs['lut_functions']

def check_power_op(cfg):

    base_node = cfg['inputs'][0]
    exp_node = cfg['inputs'][1]

    # Check all valid power conditions and return the appropriate CFG.

    # Operation: x^-1 -> 1/x
    if type(exp_node) == type(S.NegativeOne):
        return {
            "op": "div",
            "inputs": [
                sympy.Integer(1), 
                base_node
            ]
        }
    
    # Operation: x^n -> x * x * ... x
    # Performs tree duplication, which is cleaned up after compilation by the connectivity matrix compiler
    if ((type(base_node) == Symbol) or (type(base_node) == dict)) and type(exp_node) == sympy.Integer:
        return construct_balanced_tree([base_node] * int(exp_node), "mult", add_squarers=True)

    # Operation: n^x -> LUT(x) 
    if verify_base(base_node) and ((type(exp_node) == Symbol) or type(exp_node) == dict):
        return {
            "op": 'lut',
            "fn": lambda x: int(base_node) ** x,
            "base": str(base_node),
            "inputs": [exp_node]
        }

    raise Exception("MONARCH - Illegal power operation detected - input operands are {} of type {}".format((base_node, exp_node), (type(base_node), type(exp_node))))

def replace_exp(cfg):
    return {
        "op": "lut",
        "fn": lambda a: math.e ** a,
        "inputs": cfg['inputs']
    }

def check_sympy_operation(op, check_str):
    if type(op) == str:
        return False
    else:
        return op.__name__ == check_str

def check_monarch_operation(op, check_str):
    if type(op) != str:
        return False
    else:
        return op == check_str

def cleanup_cfg(cfg, op_map):
    # This function recurses through the tree and changes remaining 
    # sympy operations into monarch opcodes.
    # TODO combine the recursive CFG function cores into one, as a lot
    # of this code is duplicated below in the CFG optimisation function.
    
    if type(cfg) != dict:
        # We've reached a terminal node aka an atomic.
        return cfg

    if type(cfg['op']) != str:
        # This is a sympy operation.
        op_name = get_operation_name(cfg['op'])
        if op_name in op_map.keys():
            cfg['op'] = op_map[op_name]
            return cfg
        else:
            return None
    
    return_vals = []    
    for child in cfg['inputs']:
        return_vals.append(cleanup_cfg(child, op_map))

    for i, item in enumerate(return_vals):
        if item is not None:
            cfg['inputs'][i] = item
            return cfg

def optimise_cfg(cfg, mode=""):
    # This method recursively modifies the target CFG
    # and performs platform specific symbolic optimisations,
    # along with cleaning up some sympy-specific interpretations
    # of equation syntax e.g. Pow(x, -1) => Div(1, x)
    # Returns none when no valid optimisations can be performed on the graph.
    
    if mode != 'digital-pipelined':
        raise Exception("MONARCH - No modes of CFG optimisation other than digital-pipelined are currently supported.")

    # Recursion termination criteria
    if type(cfg) != dict:
        return None
    
    # This optimisation removes the definition of division as multiplication
    # with exponentiation of -1, and binarises the operations.
    if check_sympy_operation(cfg['op'], "Mul") or check_monarch_operation(cfg['op'], 'mult'):

        if len(cfg['inputs']) >= 2:
            nom_nodes = []
            denom_nodes = []
            for i, input_node in enumerate(cfg['inputs']):
                check = check_divided_var(input_node)
                if check is not None:
                    denom_nodes.append(check)
                else:
                    nom_nodes.append(input_node)

            noms = len(nom_nodes)
            denoms = len(denom_nodes)
            if denoms > 0:
                if noms > 1:
                    nom_tree = construct_balanced_tree(nom_nodes, 'mult')
                else:
                    nom_tree = nom_nodes[0]
                
                if denoms > 1:
                    denom_tree = construct_balanced_tree(denom_nodes, 'mult')
                else:
                    denom_tree = denom_nodes[0]
                
                return {
                    "op": "div",
                    "inputs": [nom_tree, denom_tree]
                }

    # The test below binarises a multiplication operation with more than 2 inputs.
    # It doesn't work for division, which is covered above.
    if check_sympy_operation(cfg['op'], "Mul") or check_monarch_operation(cfg['op'], 'mult'):
        if len(cfg['inputs']) > 2:
            tree = construct_balanced_tree(cfg['inputs'], opcode="mult")
            return tree

    # The test below both adds subtractions to the tree in relevant places
    # and ensures that multiple operand adds cannot occur.
    if check_sympy_operation(cfg['op'], "Add"):

        if len(cfg['inputs']) >= 2:
            pos_input_nodes = []
            neg_input_nodes = []
            for input_node in cfg['inputs']:
                check = check_negative_var(input_node)
                if check is not None:
                    neg_input_nodes.append(check)
                else:
                    pos_input_nodes.append(input_node)

            adder_tree = construct_binary_adder_tree(pos_input_nodes, neg_input_nodes)
            return adder_tree

    # This checks for power operations, and expands the operation into either a multiplier tree 
    # for x^n ops, transforms x^-1 into div(1, x) for SymPy division, or adds a LUT operation for n^x ops
    if check_sympy_operation(cfg['op'], "Pow"):
        ret_tree = check_power_op(cfg)
        if ret_tree is not None:
            return ret_tree
    
    # This checks for exponetiation (e^x) and replaces it with a LUT
    if check_sympy_operation(cfg['op'], "exp"):
        return replace_exp(cfg)

    return_vals = []    
    for child in cfg['inputs']:
        return_vals.append(optimise_cfg(child, mode=mode))

    for i, item in enumerate(return_vals):
        if item is not None:
            cfg['inputs'][i] = item
            return cfg
        
    return None     

def modify_variable(cfg, var_str, new_var_str):
    # Finds a variable by string name and substitutes it with a new symbol.     
    
    for i, input_node in enumerate(cfg['inputs']):
        if type(input_node) != dict:
            # Terminal input symbol node found.
            if str(input_node) == var_str:
                cfg['inputs'][i] = Symbol(new_var_str)
        else:
            cfg['inputs'][i] = modify_variable(cfg['inputs'][i], var_str, new_var_str)

    return cfg

def substitute_subeqs(equations):
    # Substitutes subequations defined in the system at multiple levels.

    # Preprocess equations
    system_eqs = []
    aux_eqs = []
    for lhs, rhs in equations:
        if lhs.find('/dt') != -1:
            # This equation is a system equation.
            system_eqs.append([
                lhs, 
                sympy.sympify(rhs)
            ])
        else:
            # This equation is an auxiliary equation to be substituted.
            aux_eqs.append([
                Symbol(lhs),
                sympy.sympify(rhs)
            ])
    
    subs = True
    while subs:
        subs = False
        for i, (_, rhs) in enumerate(system_eqs):
            atoms = rhs.atoms(Symbol)
            for var, lhs in aux_eqs:
                if var in atoms:
                    subs = True
                    rhs = rhs.subs(var, lhs)
            system_eqs[i][1] = rhs

    return system_eqs

def eq_to_cfg(eq): 

    equations = extract_equality_expr(eq)
    variables = []
    eq_system = []

    system_eqs = substitute_subeqs(equations)
    
    for lhs, eq_symb in system_eqs:
        
        # Recursively build computational flow graph for each
        # equation.
        cfg = get_operation(eq_symb)

        # Optimise the CFG for mode of operation.
        new_cfg = cfg
        opt_passes = 0
        while new_cfg is not None:
            cfg = new_cfg
            new_cfg = optimise_cfg(cfg, mode="digital-pipelined")
            opt_passes += 1
        
        print("MONARCH - Computational flow graph optimisation for {} complete. Cycles: {}".format(lhs, opt_passes-1))

        # Cleanup the graph by checking operations and substituting SymPy expressions
        # for monarch opcodes.
        # Open architecture database. TODO Combine into utility function with other instance of the code below
        script_dir = os.path.dirname(__file__)
        rel_path = "arch_dbs.json"
        abs_file_path = os.path.join(script_dir, rel_path)
        
        with open(abs_file_path) as file:
            dbs = json.loads(file.read())
            dbs = dbs['opcodes']
        
        op_map = {}
        for instr in dbs:
            sympy_name = dbs[instr]["sympy_op"]
            op_map[sympy_name] = str(instr)

        cfg = cleanup_cfg(cfg, op_map)
        if cfg is None:
            raise Exception("MONARCH - CFG contains unsupported operations.")

        variable = verify_differential(lhs)
        variables.append(variable)
        eq_system.append(cfg)

    # Rename variables to *_pre to reflect digitisation of the ODE.
    for i, eq in enumerate(eq_system):
        for var in variables:
            eq_system[i] = modify_variable(eq, str(var), str(var) + "_pre")

    # Rename numerical constants to *_const
    for i, eq in enumerate(eq_system):
        terminal_nodes = extract_source_nodes(eq)
        for node in terminal_nodes:
            if node.is_number or str(node) == "0":
                eq_system[i] = modify_variable(eq, str(node), str(node) + "_const")

    # Add multiplication by dt to each tree to perform Euler's method.
    for i in range(len(eq_system)):

        eq_system[i] = {
            "op": "mult",
            "inputs": [
                eq_system[i],
                Symbol("dt")
            ]
        }

        eq_system[i] = {
            "op": "add",
            "inputs": [
                eq_system[i],
                Symbol(str(variables[i]) + "_pre")
            ]
        }

    # Add top level wrapper to indicate output (root) node of the tree.
    for i, eq in enumerate(eq_system):
        eq_system[i] = {
            "root": Symbol(str(variables[i]) + "_post"),
            "cfg": eq
        }
    
    return eq_system