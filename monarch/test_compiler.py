from parsers.report_utils import report_utilisation
from parsers.equation_parse import eq_to_cfg
from parsers.cfg_compiler import cfg_to_pipeline
from simulators.simulators import pipeline_eumulator

if __name__ == "__main__":

    
    #test_equ = """
    #dx/dt = sigma * (y - x)
    #dy/dt = x * (rho - z) - y
    #dz/dt = x * y - b * z
    #"""

    """
    # Emulate the system
    args = {
        "b": 8.0/3.0,
        "sigma": 10,
        "rho": 28
    }

    init_state = {
        "x": 1,
        "y": 0,
        "z": 0
    }
    """

    test_equ = """
    dx/dt = x * (1 - x) + f * ((q - x) / (q + x)) * y
    dy/dt = x - y
    """

    init_state = {
        "x": 30,
        "y": 16
    }

    args = {
        "epsilon": 10.0 ** -2,
        "q": 10.0 ** -4,
        "f": 0.0001
    }

    compiled_cfg = eq_to_cfg(test_equ)
    pipelined_cfg = cfg_to_pipeline(compiled_cfg)

    report_utilisation(pipelined_cfg)

    #pipelined_cfg.show_report()

    

    pipeline_eumulator(
        test_equ, 
        pipelined_cfg, 
        init_state, 
        args,
        sim_time=5
    )