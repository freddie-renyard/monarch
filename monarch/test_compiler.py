from parsers.hardware_objs import HardwareUnit, CFGU, ManycoreUnit, Tile
from parsers.equation_parse import eq_to_cfg
from parsers.cfg_compiler import cfg_to_pipeline
from simulators.simulators import pipeline_eumulator, simulate_system
from simulators.asip_emulator import emulate_core
import numpy as np
from copy import copy

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
        "f": 0.0001,
        "q": 10.0 ** -4
    }

    return test_equ, args, init_state

def hodgkin_huxley():

    test_equ = """
    dv/dt = -(I_K + I_Na + I_L - I_in) / Cm 
    dn/dt = a_n * (1 - n) - b_n * n
    dm/dt = a_m * (1 - m) - b_m * m
    dh/dt = a_h * (1 - h) - b_h * h

    I_K = g_K * (n ^ 4) * (v - V_K)
    I_Na = g_Na * (m ^ 3) * h * (v - V_Na)
    I_L = g_L * (v - V_L)

    a_n = 0.01 * (10 - v) / (e ^ (1 - 0.1 * v) - 1)
    b_n = 0.125 * e ^ (-v / 80)

    a_m = 0.1 * (25 - v) / (e ^ (2.5 - 0.1 * v) - 1)
    b_m = 4 * e ^ (-v / 18)

    a_h = 0.07 * (e ^ (-v / 20))
    b_h = 1 / (e ^ (3 - 0.1 * v) + 1)
    """

    args = {
        "g_K": 36.0,
        "g_Na": 120.0,
        "g_L": 0.3,
        "V_K": -12.0,
        "V_Na": 115.0,
        "V_L": 10.613,
        "Cm": 1.0,
        "I_in": 10
    }

    init_state = {
        "v": 0.0,
        "n": 0.31767691,
        "m": 0.05293249,
        "h": 0.59612075
    }

    instances = 10
    sys_data = {
        "g_K": 36.0,
        "g_Na": 120.0,
        "g_L": 0.3,
        "V_K": -12.0,
        "V_Na": 115.0,
        "V_L": 10.613,
        "Cm": 1.0,
        "I_in": 40,
        
        "v": 0.0,
        "n": 0.31767691,
        "m": 0.05293249,
        "h": 0.59612075,

        "dt": 1.0/128.0
    }

    return test_equ, args, init_state, sys_data, instances

def lorenz_attractor():

    test_equ = """
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - b * z
    """
    args = {
        "b": 8.0/3.0,
        "sigma": 10,
        "rho": 10
    }

    init_state = {
        "x": 1.0,
        "y": 0,
        "z": 0
    }

    instances = 10
    sys_data = {
        "x": 1.0,
        "y": 0.0,
        "z": 0.0,
        "b": 8.0/3.0, # np.random.uniform(4.0/3.0, 16.0/3.0, instances),
        "sigma": 10.0, # np.random.uniform(5, 50, instances),
        "rho": 10.0, #np.random.uniform(5, 50, instances),
        "dt": 1.0/128.0
    }

    return test_equ, args, init_state, sys_data, instances

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

    instances = 10
    sys_data = {
        "v_m_2": 100.0,
        "v_m_3": 700.0,
        "k_2": 1.0,
        "k_r": 15.0,
        "k_a": 2.5,
        "k_f": 0,
        "k": 8.0,
        "b": 1.0,
        "x": np.random.uniform(10.0, 20.0, instances),
        "y": np.random.uniform(10.0, 20.0, instances)
    }

    return test_equ, args, init_state, sys_data, instances

def duplicate_test():

    test_equ = """
        dx/dt = (x + y) * 0.5
        dy/dt = (x - y)
    """

    args = {
        
    }

    init_state = {
        "x": 0.5,
        "y": 0.6
    }

    return test_equ, args, init_state

if __name__ == "__main__":

    test_equ, args, init_state, sys_data, instances = cicr()

    dt = 1.0/128.0
    compiled_cfg = eq_to_cfg(test_equ)
    pipelined_cfg = cfg_to_pipeline(compiled_cfg)

    hardware_unit = ManycoreUnit(pipelined_cfg, args, init_state, dt=dt)

    tile = Tile(
        hardware_unit,
        instances,
        sys_data,
        init_state.keys()
    )

    emulate_core()

    simulate_system(
        test_equ, sys_data, sim_time=20, dt=1.0/128.0
    )
