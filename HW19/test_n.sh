#!/bin/bash
nvcc HW19.cu -o temp -lglut -lGLU -lGL

# Define the range of n values to test (100 to 100000, step 100)
for ((n=1000; n<=1000; n+=1000))
do
    echo "Testing n=$n" >> results.txt
    
    # Run each n value 3 times
    for run in {1..3}
    do
        echo "  Run $run with n=$n"
        echo "  Run $run:" >> results.txt
        ./temp $n 0 | tee -a results.txt
        echo "  ----------" >> results.txt
    done
    
    echo "========================================" >> results.txt
done