#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include "CudaSafeCall.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define Nx 2048
#define Ny 2048
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define dx 1.0f
#define dy 1.0f
#define D_u 1.0f
#define D_v 0.0f
#define dt 0.01f

#define a 0.1f
#define c 1.0f
#define epsilon 0.012f
#define b 0.5f
#define d 1.0f
#define delta 0.0f


__global__  //Identifier for device kernel
void vec_add(float* out_u, float* out_v, float* in_u, float* in_v) {//Shape of out: [ | | | ], shape of in: [ | | | ]

	//Let's redesign this for a numerical derivative stencil. We won't bother with a dx, just assume its 1/4 for now, what we want is the difference formula. To do this, we need shared memoryfor the indices:

	//We initialize a local array that's the size of the block... plus 2? We do this to pad the outer edges so that we don't go out of bounds. These are the boundary conditions!

	// Shape of temp: [ | | | | | ]

	//Now we need to read out the index of where we are:

	int glob_ind_x = threadIdx.x + blockIdx.x * blockDim.x; //The block id is 0 so this doesn't matter but good habit (glob_ind = [0, 99])
	int glob_ind_y = threadIdx.y + blockIdx.y * blockDim.y;
	int global_index = glob_ind_x + Nx * glob_ind_y;

	int N = glob_ind_y < Ny - 1 ? glob_ind_x + Nx * (glob_ind_y + 1) : glob_ind_x + Nx * (glob_ind_y - 1);
	int S = glob_ind_y > 0 ? glob_ind_x + Nx * (glob_ind_y - 1) : glob_ind_x + Nx * (glob_ind_y + 1);
	int W = glob_ind_x > 0 ? (glob_ind_x - 1) + Nx * glob_ind_y : (glob_ind_x + 1) + Nx * glob_ind_y;
	int E = glob_ind_x < Nx - 1 ? (glob_ind_x + 1) + Nx * glob_ind_y : (glob_ind_x - 1) + Nx * glob_ind_y;

	float laplacian_u = D_u * (1 / (dx * dx)) * (in_u[E] - 2.0f * in_u[global_index] + in_u[W]) +
		D_u * (1 / (dy * dy)) * (in_u[N] - 2.0f * in_u[global_index] + in_u[S]);

	float laplacian_v = D_v * (1 / (dx * dx)) * (in_v[E] - 2.0f * in_v[global_index] + in_v[W]) +
		D_v * (1 / (dy * dy)) * (in_v[N] - 2.0f * in_v[global_index] + in_v[S]);

	float rxn_u = in_u[global_index] * (a - in_u[global_index]) * (in_u[global_index] - c) - in_v[global_index];

	float rxn_v = epsilon * (b * in_u[global_index] - d * in_v[global_index] - delta);




	out_u[global_index] = in_u[global_index] + (dt * laplacian_u) + (dt * rxn_u);
	out_v[global_index] = in_v[global_index] + (dt * laplacian_v) + (dt * rxn_v);

}

void print2DGnuplot(float* u, const char* filename) {

	int i, j, idx;

	//Print data
	FILE* fp1;
	fp1 = fopen(filename, "w+");

	// Notice we are not saving the ghost points
	for (j = 0; j < Ny; j++) {
		for (i = 0; i < Nx; i++) {
			idx = i + Nx * j;
			fprintf(fp1, "%d\t %d\t %12.16f\n", i, j, (float)u[idx]);
		}
		fprintf(fp1, "\n");
	}

	fclose(fp1);

	printf("2D GNU format data file created\n");

}

int main() {

	float* u, * v, * out_u, * out_v;

	float* d_u, * d_v, * d_out_u, * d_out_v;

	const char* filename = "fhn1.dat";

	float radius = 400.0f;

	float amplitude_u = 5.0f;
	float amplitude_v = -0.3f;

	int T_it = 20000;





	CudaSafeCall(cudaMalloc((void**)&d_u, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_v, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_out_u, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_out_v, sizeof(float) * Nx * Ny));

	u = (float*)malloc(sizeof(float) * Nx * Ny);
	v = (float*)malloc(sizeof(float) * Nx * Ny);
	out_u = (float*)malloc(sizeof(float) * Nx * Ny);
	out_v = (float*)malloc(sizeof(float) * Nx * Ny);


	memset(u, 0.0f, Nx * Ny * sizeof(float));
	memset(v, 0.0f, Nx * Ny * sizeof(float));
	for (int j = 0; j < Ny; j++) {
		for (int i = 0; i < Nx; i++) {

			int glob_ind = i + Nx * j;
			float x0 = (float)i * dx - 0.5f * dx * Nx;
			float y0 = (float)j * dy - 0.5f * dy * Ny;

			if ((x0 * x0 + y0 * y0) <= radius) {
				u[glob_ind] = amplitude_u;
				v[glob_ind] = amplitude_v;
			}

			if (((x0 - 20.0f) * (x0 - 20.0f) + (y0 - 20.0f) * (y0 - 20.0f)) <= 150.0f*radius) {
				u[glob_ind] = 1.5f;
				v[glob_ind] = 0.2f;
			}
		}
	}

	CudaSafeCall(cudaMemcpy(d_u, u, sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_v, v, sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice));

	dim3 BLOCK2D = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 GRID2D = dim3(Nx / BLOCK_SIZE_X, Ny / BLOCK_SIZE_Y, 1);

	vec_add << <GRID2D, BLOCK2D >> > (d_out_u, d_out_v, d_u, d_v); // We call this on 1 block and (Nx, Ny, 1) threads
	CudaCheckError();
	CudaSafeCall(cudaMemcpy(d_u, d_out_u, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));
	CudaSafeCall(cudaMemcpy(d_v, d_out_v, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));


	float totaltime = 0.f;
	float elapsedTime;

	for (int i = 0; i < T_it; i++) {

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Declare kernel (CUDA function)

		vec_add << <GRID2D, BLOCK2D >> > (d_out_u, d_out_v, d_u, d_v); // We call this on 1 block and (Nx, Ny, 1) threads
		CudaCheckError();
		CudaSafeCall(cudaMemcpy(d_u, d_out_u, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));
		CudaSafeCall(cudaMemcpy(d_v, d_out_v, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));



		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		totaltime += elapsedTime;
		printf("Time: %f  ms\n", elapsedTime);
	}

	printf("%f \n", (totaltime / T_it));


	CudaSafeCall(cudaMemcpy(out_u, d_out_u, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost));

	print2DGnuplot(out_u, filename);

	cudaFree(d_u); //Free the enslaved memory!!!!
	cudaFree(d_v);
	cudaFree(d_out_u); //Free the enslaved memory!!!!
	cudaFree(d_out_v);
	free(u);
	free(v);
	free(out_u);
	free(out_v);

	return 0;

}