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

// Defines
// None

// Globals
int NumberOfRandomSteps = 10'000; // Number of steps in the random walk

// Function prototypes
int getRandomDirection();
int main(int, char**);

int getRandomDirection()
{
	// Randomly return -1 or 1
	return (rand() % 2 == 0) ? -1 : 1;
}

int main(int argc, char** argv)
{
	// Seed the random number generator
	srand(time(NULL));

	// Initialize the position
	int position = 0;

	// Loop through the number of steps
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		// Call the randomWalk function to get a step
		position += getRandomDirection();
	}

	// Print the final position
	printf("Final position: %d\n", position);
	
	return 0;
}