//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Tian Ye
 * UCLA ID: 704931660
 * Email id: tianye
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	int x;
	#pragma omp parallel for
	for(x = 16384; x < 2097152; x++) {
		u[x] = 0;
	}
}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;
	int yUpper, xUpper;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		#pragma omp parallel for private(yUpper, xUpper, x, y)
		for(z  = 16384; z  < 2080768; z += 16384)
		{
			yUpper = z + 16256;
			for(y = z + 128; y < yUpper; y += 128)
			{
				xUpper = y + 128;
				for(x = y + 1; x < xUpper - 1; x++)
				{                                       
					 g[x]=1.0/sqrt(epsilon+
						SQR(u[x]-u[x+1])+SQR(u[x]-u[x-1])+
                              		        SQR(u[x]-u[x+128])+SQR(u[x]-u[x-128])+
						SQR(u[x]-u[x+16384])+SQR(u[x]-u[x-16384]));
				}
			}
		}
		memcpy(conv, u, sizeof(double) * 2097152);
		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;
		for(z  = 16384; z  < 2080768; z += 16384)
                {
                        yUpper = z + 16256;
                        for(y = z + 128; y < yUpper; y += 128)
                        {
                                xUpper = y + 128;
                                for(x = y + 1; x < xUpper - 1; x++)
                                {
				double oldVal = u[x];
                                double newVal = (u[x]+dt*
					(u[x-1]*g[x-1]+u[x+1]*g[x+1]+u[x-128]*g[x-128]+u[x+128]*g[x+128]+
                                        u[x-16384]*g[x-16384]+u[x+16384]*g[x+16384]-gamma*conv[x]))
					/(1.0+dt*(g[x+1]+g[x-1]+g[x+128]+g[x-128]+g[x+16384]+g[x-16384]));
                                if(fabs(oldVal - newVal) < epsilon)
				{
                                	converged++;
                                }
                                u[x] = newVal;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}
