# Hybrid CPU-GPU parallel programming of an N-body problem with massive bodies and test-particles


## Structure of the program :
The code is divided into several folders :
- inputs : contains the input files
- outputs : contains the output files (there is a template for each type of output file inside the folder)
- headers : contains the header codes
- sources : contains the source codes
- objects : contains the files needed during the compilation

display.py can be used to display the output data


## Compilation and execution instructions :
To compile the code : make
To run the simulation : ./run (input file path)

/!\ : Please specify the OS used at the beginning of the Makefile to use the following features. 

## Cleaning options
To remove object files and executables : make clean
To remove output files : make clean_outputs
To remove all generated files (object files, executables and outputs) : make clean_all

