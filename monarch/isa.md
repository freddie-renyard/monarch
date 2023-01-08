## MONArch Implementations

This is a summary of the approaches that have been developed in this repository. Refactoring is needed to unify the API 
and to integrate the methods into one toolchain.

# Common features

All of the architectures model first order dynamics and they use Euler's method for numerical integration.

# Cellular Implementation

This involves compiling the equations into a cellular phase space representation, and looking up the 
correct vector component on each simulation cycle. It has the advantage of being very fast, but has
large space complexity and high BRAM utilisation when deployed to FPGA, especially when the dimensionality
of the system is above 2.

A compression with analysis of the system vectors with the k-means algorithm and compiling a pointer space
was also implemented, which allowed for better tradeoffs with memory utilisation and accuracy.

# Pipeline Synthesis Implementation

This approach involves compiling the equations into their binary tree representation, as supported by 
the hardware, followed by compilation to adjacency matrix representation. The software uses the architecture
database to add appropriate register delays to ensure that data can be inputted into the pipeline on every 
clock cycle.

The approach is very hardware intensive, but has extremely high throughput and is better used for smaller
models. Also, in models where data cannot be processed on every cycle e.g. networked models with
high connectivity, the extra hardware overhead carries no advantage.

# Manycore implementation

This approach is an extension of the above approach, but abstracts the tree structure into software in
order to reduce the amount of arithmetic hardware that is synthesised. The tree structure is analysed in 
adjacency matrix form, and so the compiler toolchain is shared with the pipeline synthesis method. 

A certain number of cores are allocated to each tile. The compiler steps through the matrix and 
allocates arithmetic instructions on a round robin basis, compiling into the assembly language detailed 
below. This is then compiled to machine code that can be natively executed on the hardware target. 

Parallelism can be implemented in three ways:

1. Increasing core count within the system
2. Increasing ALUs to compute other instances of the same model in parallel.
3. Increasing tile count, which can be used to model different systems.

The main computational units are:

1. Tiles - These compute whole systems
2. Columns - These compute the same system on a different register set, and therefore different model instances
3. Cores - These compute different parts of the computational graph of a system

Increasing core count has diminishing returns after a certain period, depending on graph structure. 
Increasing columns can be used to perform an efficient space time tradeoff. The program memories are shared
between each of the columns, which reduces BRAM utilisation.

# Manycore ISA

Each core consists of one program memory block and program counter, and the following associated hardware:

1. Input and working registers (24 registers)
2. Output registers (8 registers)
3. ALU/s
4. Stall counter
5. CSRs, including exception data, etc.
6. Constant pool

Instructions are encoded as follows:

- `Bits [4:0] - Opcode`
- `Bits [8:4] - Input register 1`
- `Bits [13:9] - Input register 2`
- `Bits [18:14] - Output register`

This produces a 20-bit instruction. This can be modified for larger register maps.

# Opcodes

| Instruction Function           | Assembly  | Instruction Type | Encoding| Notes |
| -------------------------------|:---------:| ----------------:| -------:| -----:|
| Multiplication with registers  | `mult`    | `0b00000`        | `3r`    |       |
| Addition with registers        | `add`     | `0b00001`        | `3r`    |       |
| Subtraction with registers     | `sub`     | `0b00010`        | `3r`    |       |
| Division with registers        | `div`     | `0b00011`        | `3r`    |       |
| Lookup with registers          | `lut`     | `0b00100`        | `2rl`   | The second operand encodes the table to use.|
| Multiplication with constant   | `mult_c`  | `0b00000`        | `2rc`   |       |
| Addition with constant         | `add_c`   | `0b00001`        | `2rc`   |       |
| Subtraction with constant      | `sub_c`   | `0b00010`        | `2rc`   |       |
| Division with constant         | `div_c`   | `0b00011`        | `2rc`   |       |
| No operation                   | `nop`     | `0b11111`        | `0r`    | The first operand encodes the number of cycles to stall |

The lookup table block in the ALU uses its second argument to select the table.

| Lookup Table Function | Encoding |
| ----------------------|:---------|
| `e ^ x`               | `0b00000`|

# CSRs

One 32 bit CSR is implemented for flags.
- Bit 0 - Division by 0
- Bit 1 - Overflow in mult
- Bit 2 - Overflow in add
- Bit 3 - Overflow in sub
- Bit 4 - Overflow in div

# Constant Pool

The constant pool is a pool of 32 32-bit integers that is used for constants that are common to
the whole system. This includes multiplication by fractions, 1 or -1, or the timestep value (dt).