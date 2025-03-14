// Name: Kyle Earp
// Two body problem
// nvcc HW17.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some not so crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
#define NUMBER_OF_SPHERES 50

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
float4 Position[NUMBER_OF_SPHERES], Velocity[NUMBER_OF_SPHERES], Force[NUMBER_OF_SPHERES], Colors[NUMBER_OF_SPHERES];
float Mass[NUMBER_OF_SPHERES];

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	
	int yeahBuddy, placementAttempts, maxAttempts = 1000;
	float dx, dy, dz, seperation;

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		yeahBuddy = 0;
		placementAttempts = 0;
		while(yeahBuddy == 0)
		{
			Position[i].x = (LENGTH_OF_BOX - DIAMETER)*(float)rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Position[i].y = (LENGTH_OF_BOX - DIAMETER)*(float)rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Position[i].z = (LENGTH_OF_BOX - DIAMETER)*(float)rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			
			yeahBuddy = 1;
			for(int j = 0; j < i; j++)
			{
				dx = Position[i].x - Position[j].x;
				dy = Position[i].y - Position[j].y;
				dz = Position[i].z - Position[j].z;
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				if(seperation < DIAMETER) 
				{
					placementAttempts++;
					if(placementAttempts > maxAttempts)
					{
						printf("%d spheres could not fit in the box. Failed at sphere indexed %d, try again\n", NUMBER_OF_SPHERES, i);
						exit(0);
					}
					yeahBuddy = 0;
					break;
				}
			}
		}
		
		Velocity[i].x = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Velocity[i].y = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Velocity[i].z = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		
		Colors[i].x = (float)rand()/RAND_MAX;
		Colors[i].y = (float)rand()/RAND_MAX;
		Colors[i].z = (float)rand()/RAND_MAX;

		Force[i].x = 0.0;
		Force[i].y = 0.0;
		Force[i].z = 0.0;

		Mass[i] = 1.0;
	}
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		glColor3d(Colors[i].x,Colors[i].y,Colors[i].z);
		glPushMatrix();
		glTranslatef(Position[i].x, Position[i].y, Position[i].z);
		glutSolidSphere(radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(Position[i].x > halfBoxLength)
		{
			Position[i].x = 2.0*halfBoxLength - Position[i].x;
			Velocity[i].x = - Velocity[i].x;
		}
		else if(Position[i].x < -halfBoxLength)
		{
			Position[i].x = -2.0*halfBoxLength - Position[i].x;
			Velocity[i].x = - Velocity[i].x;
		}
		
		if(Position[i].y > halfBoxLength)
		{
			Position[i].y = 2.0*halfBoxLength - Position[i].y;
			Velocity[i].y = - Velocity[i].y;
		}
		else if(Position[i].y < -halfBoxLength)
		{
			Position[i].y = -2.0*halfBoxLength - Position[i].y;
			Velocity[i].y = - Velocity[i].y;
		}
				
		if(Position[i].z > halfBoxLength)
		{
			Position[i].z = 2.0*halfBoxLength - Position[i].z;
			Velocity[i].z = - Velocity[i].z;
		}
		else if(Position[i].z < -halfBoxLength)
		{
			Position[i].z = -2.0*halfBoxLength - Position[i].z;
			Velocity[i].z = - Velocity[i].z;
		}
	}
}

void get_forces()
{
	float4 d, dv;
	float r, r2, forceMag, inout;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		Force[i].x = 0.0;
		Force[i].y = 0.0;
		Force[i].z = 0.0;
	}

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for(int j = i+1; j < NUMBER_OF_SPHERES; j++)
		{
			d.x = Position[j].x - Position[i].x;
			d.y = Position[j].y - Position[i].y;
			d.z = Position[j].z - Position[i].z;

			r2 = d.x*d.x + d.y*d.y + d.z*d.z;
			r = sqrt(r2);
			
			forceMag =  Mass[i]*Mass[j]*GRAVITY/r2;

			if (r < DIAMETER)
			{
				dv.x = Velocity[j].x - Velocity[i].x;
				dv.y = Velocity[j].y - Velocity[i].y;
				dv.z = Velocity[j].z - Velocity[i].z;
				inout = d.x*dv.x + d.y*dv.y + d.z*dv.z;
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}

			Force[i].x += forceMag*d.x/r;
			Force[i].y += forceMag*d.y/r;
			Force[i].z += forceMag*d.z/r;
			Force[j].x -= forceMag*d.x/r;
			Force[j].y -= forceMag*d.y/r;
			Force[j].z -= forceMag*d.z/r;
		}
	}
}

void move_bodies(float time)
{
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(time == 0.0)
		{
			Velocity[i].x += 0.5*DT*(Force[i].x - DAMP*Velocity[i].x)/Mass[i];
			Velocity[i].y += 0.5*DT*(Force[i].y - DAMP*Velocity[i].y)/Mass[i];
			Velocity[i].z += 0.5*DT*(Force[i].z - DAMP*Velocity[i].z)/Mass[i];
		}
		else
		{
			Velocity[i].x += DT*(Force[i].x - DAMP*Velocity[i].x)/Mass[i];
			Velocity[i].y += DT*(Force[i].y - DAMP*Velocity[i].y)/Mass[i];
			Velocity[i].z += DT*(Force[i].z - DAMP*Velocity[i].z)/Mass[i];
		}

		Position[i].x += DT*Velocity[i].x;
		Position[i].y += DT*Velocity[i].y;
		Position[i].z += DT*Velocity[i].z;
	}

	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}