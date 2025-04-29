// Name: Kyle Earp
// GPU random walk. 
// nvcc HW28.cu -o temp

/*
 What to do:
 This is some code that runs a random walk for 10000 steps.
 Use cudaRand and run 10 of these runs at once with diferent seeds on the GPU.
 Print out all 10 final positions.
*/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

// Defines
#define BLOCK_SIZE 10 // Number of threads per block

// Globals
int NumberOfRandomSteps = 10'000; // Number of steps in the random walk
int NumberOfRandomWalks = 10; // Number of random walks to perform
int *positions_CPU; // CPU pointer for positions
int *positions_GPU; // GPU pointer for positions
dim3 BlockSize, GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
__global__ void randomWalk(int *, int, int, unsigned long long);
void cleanUp();
int main(int, char**);

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

void setUpDevices()
{
	// Set up the block and grid sizes
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;

	GridSize.x = (int)((NumberOfRandomWalks - 1) / BlockSize.x) + 1; // Number of blocks needed
	GridSize.y = 1;
	GridSize.z = 1;
}

void allocateMemory()
{
	// Allocate memory on the CPU for positions
	positions_CPU = (int*)malloc(NumberOfRandomWalks * sizeof(int));

	// Allocate memory on the GPU for positions
	cudaMalloc(&positions_GPU, NumberOfRandomWalks * sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void randomWalk(int *positions, int numberOfSteps, int n, unsigned long long timeValue)
{
	// Get the thread ID
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId < n)
	{
		// Initialize the position
		int position = 0;

		curandState state;
		curand_init(threadId, timeValue, 0, &state); // Initialize the random state with a seed

		for(int i = 0; i < numberOfSteps; i++)
		{
			// Get a random direction using curand
			position += 2*(curand(&state) % 2) - 1; // Randomly return -1 or 1
		}

		// Store the final position in the array
		positions[threadId] = position;
	}
}

void cleanUp()
{
	// Free the GPU memory
	cudaFree(positions_GPU);
	cudaErrorCheck(__FILE__, __LINE__);

	// Free the CPU memory
	free(positions_CPU);
}

int main(int argc, char** argv)
{
	// Set up the devices
	setUpDevices();

	// Allocate memory for positions
	allocateMemory();

	// Launch the kernel to perform random walks
	randomWalk<<<GridSize, BlockSize>>>(positions_GPU, NumberOfRandomSteps, NumberOfRandomWalks, (unsigned long long)time(NULL));
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy the results back to the CPU
	cudaMemcpy(positions_CPU, positions_GPU, NumberOfRandomWalks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize(); // Wait for the GPU to finish
	cudaErrorCheck(__FILE__, __LINE__);

	// Print the final positions and calculate the average position
	int averagePosition = 0;
	for(int i = 0; i < NumberOfRandomWalks; i++)
	{
		averagePosition += positions_CPU[i];
		printf("Final position of walk %d: %d\n", i, positions_CPU[i]);
	}
	averagePosition /= NumberOfRandomWalks;
	printf("Average position of all walks: %d\n", averagePosition);
	
	// Clean up memory
	cleanUp();

	return 0;
}