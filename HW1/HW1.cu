// Vector addition on the CPU
// Name: Kyle Earp
// nvcc HW1.cu -o temp
/*
 What to do:
 1. Understand every line of code and be able to explain it in class.
 2. Compile, run, and play around with the code.
*/

// Include files
#include <sys/time.h> //header file with functions dealing with time
#include <stdio.h> // header file with functions dealing with input/output

// Defines
#define N 10000 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; // global pointers to dynamically allocated arrays
float Tolerance = 0.00000001; // tolerance for float sums

// Function prototypes
void allocateMemory(); // function to allocate memory for the three global arrays
void initialize(); // function to initialize the three global arrays
void addVectorsCPU(float*, float*, float*, int); // function to add two vectors on the CPU
int  check(float*, int); // function to check the results of the vector addition
long elaspedTime(struct timeval, struct timeval); // function to calculate the elapsed time
void CleanUp(); // function to deallocate memory

//Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float)); //dynamically allocates memory for an N sized array of floats
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
}

//Loading values into the vectors that we will add.
void initialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;		// initialize A_CPU with float values of i
		B_CPU[i] = (float)(2*i);	// initialize B_CPU with float values of 2*i
	}
}

//Adding vectors a and b then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = a[id] + b[id]; // add the two vectors and store the result in c
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	int id; // Initializing id
	double sum = 0.0; // Initializing sum to 0.0
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(id = 0; id < n; id++)
	{ 
		sum += c[id]; // summing up the values in c
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) // checking if the sum is close enough to 3.0*(m*(m+1))/2.0 => the exact sum of the two vectors
	{
		return(1); // return 1 if true
	}
	else 
	{
		return(0); // return 0 if false
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

//Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
}

int main()
{
	timeval start, end; // Structs to hold the start and end time.
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	initialize();

	// Starting the timer.	
	gettimeofday(&start, NULL);

	// Add the two vectors.
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);

	// Stopping the timer.
	gettimeofday(&end, NULL);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
	{
		printf("\n\n Something went wrong in the vector addition\n"); // If check returns 0 then something went wrong.
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the CPU"); // If check returns 1 then everything went correctly.
		printf("\n The time it took was %ld microseconds", elaspedTime(start, end)); // Printing out the time it took to add the two vectors.
	}
	
	// Your done so cleanup your room.	
	CleanUp();
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n");
	
	return(0);
}

