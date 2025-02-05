// Name: Kyle Earp
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
dim3 BlockSize;
dim3 GridSize;
float *Pixels_CPU;
float *Pixels_GPU;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void setUpDevices();
void allocateMemory();
void initialize();
void cudaErrorCheck(const char*, int);
__global__ void escapeOrNotColorGPU(float*, float, float, float, float);
void cleanUp();
void display(void);

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
	// Set up the block and grid sizes.
	// Each block will work on a row of pixels.
	BlockSize.x = WindowWidth;
	BlockSize.y = 1;
	BlockSize.z = 1;

	// We need a block for each column of pixels.
	GridSize.x = WindowHeight;
	GridSize.y = 1;
	GridSize.z = 1;
}

void allocateMemory()
{
	//We need the 3 because each pixel has a red, green, and blue value.
	Pixels_CPU = (float*)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float));
}

void initialize()
{
	// Make sure the array is initialized to 0.
	for(int i = 0; i < WindowWidth*WindowHeight*3; i++)
	{
		Pixels_CPU[i] = 0.0;
	}
}

__global__ void escapeOrNotColorGPU(float *pixels, float xMin, float stepSizeX, float yMin, float stepSizeY)
{
	// threadIdx.x is the index of the x coordinate, so we multiply by the step size of x and add xMin to get the x coordinate.
	float x = xMin + (threadIdx.x * stepSizeX);
	// blockIdx.x is the index of the y coordinate, so we multiply by the step size of y and add yMin to get the y coordinate.
	float y = yMin + (blockIdx.x * stepSizeY);
	// k is the index of the first element of the pixel in the pixel array.
	int k = (blockIdx.x*blockDim.x + threadIdx.x)*3;
	
	// We'll need a temporary variable to store the old value of x.
	float tempX;
	// Calculate the initial magnitude of the complex number.
	float mag = sqrt(x*x + y*y);

	// Initialize the count to 0.
	int count = 0;

	// Loop until the magnitude is greater than MAXMAG or we've reached MAXITERATIONS.
	while(mag < MAXMAG && count < MAXITERATIONS)
	{
		// Store the old value of x.
		tempX = x;
		// Calculate the new value of x.
		x = x*x - y*y + A;
		// Calculate the new value of y.
		y = (2.0 * tempX * y) + B;
		// Calculate the new magnitude.
		mag = sqrt(x*x + y*y);
		// Increment the count.
		count++;
	}
	// If we did not reach MAXITERATIONS, color the pixel black.
	if(count < MAXITERATIONS)
	{
		pixels[k] = 0.0;
		pixels[k+1] = 0.0;
		pixels[k+2] = 0.0;
	}
	// If we did reach MAXITERATIONS, color the pixel red.
	else
	{
		pixels[k] = 1.0;
		pixels[k+1] = 0.0;
		pixels[k+2] = 0.0;
	}
}

void cleanUp()
{
	// Free the memory.
	free(Pixels_CPU);
	cudaFree(Pixels_GPU);
}

void display(void) 
{
	// Set up the devices, allocate memory, and initialize the pixel array.
	setUpDevices();
	allocateMemory();
	initialize();

	// Set the step sizes for x and y.
	float stepSizeX = (XMax - XMin)/((float)WindowWidth);
	float stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	// Copy the pixel array to the GPU
	cudaMemcpyAsync(Pixels_GPU, Pixels_CPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Call the kernel
	escapeOrNotColorGPU<<<GridSize, BlockSize>>>(Pixels_GPU, XMin, stepSizeX, YMin, stepSizeY);
	cudaErrorCheck(__FILE__, __LINE__);

	// Copy the pixel array back to the CPU
	cudaMemcpyAsync(Pixels_CPU, Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);

	// Wait for the GPU to finish
	cudaDeviceSynchronize();

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, Pixels_CPU);
	glFlush();

	// Clean up the memory.
	cleanUp();

	// Flush the print buffer.
	printf("\n");
}

int main(int argc, char** argv)
{
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}