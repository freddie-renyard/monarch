from parsers.report_utils import report_utilisation
from parsers.equation_parse import eq_to_cfg
from parsers.cfg_compiler import cfg_to_pipeline

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

    compiled_cfg = eq_to_cfg(test_equ)
    pipelined_cfg = cfg_to_pipeline(compiled_cfg)

    report_utilisation(pipelined_cfg)