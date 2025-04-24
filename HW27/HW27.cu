// Name: Kyle Earp
// CPU random walk. 
// nvcc HW27.cu -o temp

/*
 What to do:
 Create a function that returns a random number that is either -1 or 1.
 Start at 0 and call this function to move you left (-1) or right (1) one step each call.
 Do this 10000 times and print out your final position.
*/

// Include files
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
// None

// Globals
int NumberOfSteps = 10'000; // Number of steps in the random walk

// Function prototypes
int randomWalk();
int main(int, char**);

int randomWalk()
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
	for(int i = 0; i < NumberOfSteps; i++)
	{
		// Call the randomWalk function to get a step
		position += randomWalk();
	}

	// Print the final position
	printf("Final position: %d\n", position);
	
	return 0;
}