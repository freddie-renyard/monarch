

def get_symbols(cfg):
    
    ret_lst = []
    for input_node in cfg['inputs']:
        if type(input_node) != dict:
            # A valid symbol has been found.
            ret_lst += [input_node]
        else:
            ret_lst += get_symbols(input_node)

    return ret_lst

def cfg_to_pipeline(eq_system):

    # Extract all graph input symbols (variables, constant-variables, and constants).
    symbols = []
    for eq in eq_system:
        print(eq)
        symbols += get_symbols(eq["cfg"])

    input_set = set(symbols)

