#!/bin/bash
# Compile the CUDA program
nvcc HW20.cu -o temp -lglut -lGLU -lGL
# Test the program with an n value and runs the program 5 times

echo "n=$1" >> results.txt

for run in {1..5}
do
    echo "  Run $run with n=$1"
    echo "  Run $run:" >> results.txt
    ./temp $1 $2 | tee -a results.txt
    echo "  ----------" >> results.txt
done