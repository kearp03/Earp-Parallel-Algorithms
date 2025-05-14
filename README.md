# Earp Parallel Algorithms
 Source Code for the Math Topics class Parallel Algorithms taken Spring 2025 with Dr. Bryant Wyatt. This course focused on 

### Topics Covered
* CUDA programming basics and syntax
* CPU vs GPU vector operations
* Memory management (allocation, transfer, cleanup)
* CUDA error checking and debugging
* Performance timing and optimization techniques
* Block and grid configurations
* CUDA device properties and capabilities
* Parallel reduction algorithms
* Dot product implementation on GPU
* Shared memory usage
* Atomic operations
* CUDA streams and concurrent execution
* Page-locked (pinned) memory
* Overlapping computation with data transfers
* Ray tracing on GPU


### Compiling and Executing Code
This repository contains mostly CUDA files, with some suplimentary bash scripts, text files, PDF files, and C files. Paper homework assignments have been recreated as PDF files. To execute the CUDA files, please use the nvcc compiler provided by NVIDIA in the following way:
```bash
# Compile the CUDA source file
nvcc source_file.cu -o executable_name

# Execute the compiled program
./executable_name
```

When a bash script is provided in a folder, this excecution in implimented in that file. First, ensure that the script has executable permissions, and then execute the script to compile and run that CUDA or C file.
```bash
# Give bash script executable permissions
chmod 755 run.sh

# Execute the bash script
./run.sh
```
You need to give executable permission only once. Afterwards you need only to execute the file.
