# Input file

# General parameters
Number of massive bodies :
Number of test-particles :        
Integration over :                # use "(number of steps) steps", "(integration time) years" or "(computation time) minutes"
Time-step :                       # in days

# Initial configurations
Initial massive bodies configuration :        # use "sun+gas_planets"
Initial test-particles configuration :        # use "random"

# Computation settings
Integration method :                 # use "leapfrog"
Integration mode :                   # use "cpu", "gpu" or "gpu_multi-step"
Number of blocks used :              # for GPU calculations
Number of threads per block :        # for GPU calculations
Number of sub-steps :                # for multi-step calculations

# File writing settings
File name suffix :                   # use "none" for no suffix, "mode" to match the integration mode or any custom string (no spaces)
Positions writing frequency :        # writes positions every "(number of steps) steps" or "(percentage of total simulation) %"
