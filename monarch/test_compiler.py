from parsers.equation_parse import eq_to_cfg

if __name__ == "__main__":

    test_equ = """
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - b * z
    """
    
    # COMPILATION STEPS:
    # 1. eq_to_cfg - this converts string equations into a CFG
    #   - Equation parsing
    #   - Platform specific optimisation
    #   - Digitisation of the ODEs
    # 2. cfg_to_pipeline - this converts cfg to pipelined connectivity matrix description
    #   - Register addition
    #   - Length optimisation
    #   - Constant register pruning
    # 3. pipeline_to_img - this converts the pipeline to hardware files, ready for verilog synthesis
    #   - .vh and .mem files

    eq_to_cfg(test_equ)