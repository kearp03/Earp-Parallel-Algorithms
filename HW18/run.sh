nvcc HW18_CPU.cu -o temp -lglut -lm -lGLU -lGL
echo "CPU"
./temp $1 $2

nvcc HW18.cu -o temp -lglut -lm -lGLU -lGL
echo "GPU"
./temp $1 $2