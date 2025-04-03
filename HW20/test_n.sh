#!/bin/bash
rm -f temp
# Compile the CUDA program
nvcc HW20.cu -o temp -lglut -lGLU -lGL -use_fast_math
# Test the program with an n value and runs the program 5 times
n=8192
echo "n=$n" >> results.txt

for run in {1..5}
do
    echo "  Run $run with n=$n"
    echo "  Run $run:" >> results.txt
    ./temp $n 0 | tee -a results.txt
    echo "  ----------" >> results.txt
done