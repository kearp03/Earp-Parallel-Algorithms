// Name: Kyle Earp
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artistic flare. 
 Don't cut off your ear or anything but make Vincent wish he had, had a GPU.
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
unsigned int WindowWidth;
unsigned int WindowHeight;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float*, float, float, float, float, int);

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

__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int windowWidth) 
{
	float x,y,mag,tempX;
	int count, id, k;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	for(id = windowWidth*blockIdx.x + threadIdx.x; id < windowWidth*(blockIdx.x + 1); id += blockDim.x)
	{
		//Getting the offset into the pixel buffer.
		//We need the 3 because each pixel has a red, green, and blue value.
		k = 3*id;

		//Assigning each thread its x and y value of its pixel in each iteration of the for loop.
		x = xMin + (id%windowWidth)*dx;
		y = yMin + blockIdx.x*dy;
		float xy = x - y;
		count = 0;
		mag = sqrt(x*x + y*y);;
		while (mag < maxMag && count < maxCount) 
		{
			//We will be changing the x but we need its old value to find y.	
			tempX = x; 
			x = x*x - y*y + A;
			y = (2.0 * tempX * y) + B;
			mag = sqrt(x*x + y*y);
			count++;
		}
		
		//Setting the color values
		if(count < maxCount) //It escaped
		{
			pixels[k]     = 0.0;
			pixels[k + 1] = 0.0;
			pixels[k + 2] = 0.0;
		}
		else //It Stuck around
		{
			// Assumes that xMin = -xMax and yMin = -yMax
			pixels[k]	 = -xy/(xMin + yMin) + 0.85;
			pixels[k + 1] = -xy/(xMin + yMin) + 0.45;
			pixels[k + 2] = -xy/(xMin + yMin) + 0.5;
		}
	}
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float *pixelsCPU, *pixelsGPU; 
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	//Threads in a block - Constant 1024
	blockSize.x = 1024; //WindowWidth;
	blockSize.y = 1;
	blockSize.z = 1;
	
	//Blocks in a grid - WindowHeight, which has a maximum of 65535
	gridSize.x = WindowHeight;
	gridSize.y = 1;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Waiting for the GPU to finish.
	cudaDeviceSynchronize();

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU);
	glFlush();

	//Cleaning up
	free(pixelsCPU);
	cudaFree(pixelsGPU);
}

// Add reshape function
void reshape(int w, int h)
{
    WindowWidth = w;
    WindowHeight = h;
    glViewport(0, 0, w, h);
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	
	// Get the screen width and height
	WindowWidth = glutGet(GLUT_SCREEN_WIDTH);
	WindowHeight = glutGet(GLUT_SCREEN_HEIGHT);
	
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
	// Add reshape function
	glutReshapeFunc(reshape);
   	glutMainLoop();
}
