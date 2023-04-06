from serial_tools.serial_tools import FPGAPort

if __name__ == "__main__":

    port = FPGAPort(
        sim_dims=2,
        instances=10,
        timesteps=1000
    )
