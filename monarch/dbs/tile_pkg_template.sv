package tile_pkg;
    localparam INSTR_WIDTH = <instr_width>;
    localparam REG_WIDTH   = <reg_width>;
    localparam DATA_WIDTH  = <data_width>;
    localparam DATA_RADIX  = <data_radix>;
    localparam N_CORES     = <n_cores>;
    localparam N_COLUMNS   = <n_columns>;
    localparam N_BANK      = <n_bank>;
    localparam N_BANK_SIZE = <n_bank_size>;
    localparam N_EXT_RD_PORTS = <n_ext_rd_ports>;
    localparam N_EXT_WR_PORTS = <n_ext_wr_ports>;  

    localparam N_RD_PORTS = 2 * N_CORES + N_EXT_RD_PORTS;
    localparam N_WR_PORTS = N_CORES + N_EXT_WR_PORTS;
endpackage