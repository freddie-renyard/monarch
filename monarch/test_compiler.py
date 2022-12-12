from parsers.report_utils import report_utilisation
from parsers.equation_parse import eq_to_cfg
from parsers.cfg_compiler import cfg_to_pipeline
from simulators.simulators import pipeline_eumulator

def oregonator_model():

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

    return test_equ, init_state, args

def lorenz_attractor():

    test_equ = """
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - b * z
    """
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

    return test_equ, args, init_state

def cicr():

    test_equ = """
        dx/dt = z_0 - z_2 + z_3 + k_f * y - k * x + z_1 * b
        dy/dt = z_2 - z_3 - k_f * y

        n = 2
        m = 2
        p = 2
        z_0 = 1
        z_1 = 6
        z_2 = v_m_2 * ((x ** n) / (k_2 ** n + x ** n))
        z_3 = v_m_3 * ((y ** m) / (k_r ** m + y ** m)) * ((x ** p) / (k_a ** p + x ** p)) 
    """

    args = {
        "v_m_2": 100.0,
        "v_m_3": 700.0,
        "k_2": 1.0,
        "k_r": 15.0,
        "k_a": 2.5,
        "k_f": 0,
        "k": 8.0,
        "b": 1.0
    }

    init_state = {
        "x": 16,
        "y": 16
    }

    return test_equ, args, init_state

def duplicate_test():

    test_equ = """
        dx/dt = (x + y) ** 3
    """

    args = {
        "y": 1
    }

    init_state = {
        "x": 1.1
    }

    return test_equ, args, init_state

if __name__ == "__main__":

    test_equ, args, init_state = duplicate_test()

    compiled_cfg = eq_to_cfg(test_equ)
    pipelined_cfg = cfg_to_pipeline(compiled_cfg)
    
    report_utilisation(pipelined_cfg)
    pipelined_cfg.remove_dup_branches()
    report_utilisation(pipelined_cfg)

    pipelined_cfg.verify_against_dbs()
    pipelined_cfg.compute_predelay()

    #pipelined_cfg.show_report()
    
    pipeline_eumulator(
        test_equ, 
        pipelined_cfg, 
        init_state, 
        args,
        sim_time=0.06
    )