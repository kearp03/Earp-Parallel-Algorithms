// Name: Kyle Earp
// Vector addition on two GPUs.
// nvcc HW23.cu -o temp
/*
 What to do:
 This code adds two vectors of any length on a GPU.
 Rewriting the Code to Run on Two GPUs:

 1. Check GPU Availability:
    Ensure that you have at least two GPUs available. If not, report the issue and exit the program.

 2. Handle Odd-Length Vector:
    If the vector length is odd, ensure that you select a half N value that does not exclude the last element of the vector.

 3. Send First Half to GPU 1:
    Send the first half of the vector to the first GPU, and perform the operation of adding a to b.

 4. Send Second Half to GPU 2:
    Send the second half of the vector to the second GPU, and again perform the operation of adding a to b.

 5. Return Results to the CPU:
    Once both GPUs have completed their computations, transfer the results back to the CPU and verify that the results are correct.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector

// Global variables
int* GPU_Ns; // Length of the vector for each GPU
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float **A_GPU, **B_GPU, **C_GPU; //GPU pointers
// float *A_GPU1, *B_GPU1, *C_GPU1, *A_GPU2, *B_GPU2, *C_GPU2; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 *GridSizes; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;
int GPUCount = 0; // Number of GPUs available

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void initialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float*, float*, float*, int);
bool  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	cudaGetDeviceCount(&GPUCount);
	printf("\n\n Number of GPUs available: %d", GPUCount);
	// if(GPUCount < 2)
	// {
	// 	printf("\n\n You do not have two GPUs. Please run on a machine with two GPUs. Loser.\n");
	// 	exit(0);
	// }

	// Allocating the GPU_Ns array to hold the number of elements for each GPU.
	GPU_Ns = (int*)malloc(GPUCount*sizeof(int));

	// Getting the number of elements for each GPU.
	int runningTotal = 0;
	for(int i = 0; i < GPUCount-1; i++)
	{
		GPU_Ns[i] = N/GPUCount;
		runningTotal += GPU_Ns[i];
	}
	GPU_Ns[GPUCount-1] = N - runningTotal;

	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// Allocating the GridSizes array to hold the grid sizes for each GPU.
	GridSizes = (dim3*)malloc(GPUCount*sizeof(dim3));
	for(int i = 0; i < GPUCount; i++)
	{
		GridSizes[i].x = (GPU_Ns[i] - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
		GridSizes[i].y = 1;
		GridSizes[i].z = 1;
	}
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	// Allocate memory for the GPU pointers
	A_GPU = (float**)malloc(GPUCount*sizeof(float*));
	B_GPU = (float**)malloc(GPUCount*sizeof(float*));
	C_GPU = (float**)malloc(GPUCount*sizeof(float*));
	
	// Allocate memory for each GPU
	for(int i = 0; i < GPUCount; i++)
	{
		cudaSetDevice(i);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMalloc(A_GPU + i,GPU_Ns[i]*sizeof(float));
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMalloc(B_GPU + i,GPU_Ns[i]*sizeof(float));
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMalloc(C_GPU + i,GPU_Ns[i]*sizeof(float));
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

// Loading values into the vectors that we will add.
void initialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n) // Making sure we are not working on memory we do not own.
	{
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	// Freeing device "GPU" memory.
	for(int i = 0; i < GPUCount; i++)
	{
		cudaSetDevice(i);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaFree(A_GPU[i]);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaFree(B_GPU[i]);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaFree(C_GPU[i]);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	initialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	for(int i = 0; i < GPUCount; i++)
	{
		cudaSetDevice(i);
		cudaErrorCheck(__FILE__, __LINE__);

		// Copy Memory from CPU to GPU
		cudaMemcpyAsync(A_GPU[i], A_CPU + i*GPU_Ns[i], GPU_Ns[i]*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B_GPU[i], B_CPU + i*GPU_Ns[i], GPU_Ns[i]*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
		// Launch the kernel on the GPU.
		addVectorsGPU<<<GridSizes[i],BlockSize>>>(A_GPU[i], B_GPU[i] ,C_GPU[i], GPU_Ns[i]);
		cudaErrorCheck(__FILE__, __LINE__);
		// Copy Memory from GPU to CPU
		cudaMemcpyAsync(C_CPU + i*GPU_Ns[i], C_GPU[i], GPU_Ns[i]*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}

	for(int i = 0; i < GPUCount; i++)
	{
		cudaSetDevice(i);
		cudaErrorCheck(__FILE__, __LINE__);
		// Making sure the GPU and CPU wait until each other are at the same place.
		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
	}

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}