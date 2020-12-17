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

#define Nx 128
#define Ny 128
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define dx 0.01f
#define dy 0.01f
#define dt 0.0001f

#define g 9.81f
#define mean_H 9.0f
#define nu 0.05f
#define b 0.003f


__global__  //Identifier for device kernel
void vec_add(float* out_u, float* out_v, float* out_h, float* in_u, float* in_v, float* in_h) {

	int glob_ind_x = threadIdx.x + blockIdx.x * blockDim.x;
	int glob_ind_y = threadIdx.y + blockIdx.y * blockDim.y;
	int global_index = glob_ind_x + Nx * glob_ind_y;

	int N = glob_ind_y < Ny - 1 ? glob_ind_x + Nx * (glob_ind_y + 1) : glob_ind_x + Nx * (glob_ind_y - 1); //No-Flux BC's
	int S = glob_ind_y > 0 ? glob_ind_x + Nx * (glob_ind_y - 1) : glob_ind_x + Nx * (glob_ind_y + 1);
	int W = glob_ind_x > 0 ? (glob_ind_x - 1) + Nx * glob_ind_y : (glob_ind_x + 1) + Nx * glob_ind_y;
	int E = glob_ind_x < Nx - 1 ? (glob_ind_x + 1) + Nx * glob_ind_y : (glob_ind_x - 1) + Nx * glob_ind_y;

	float laplacian_u = (1 / (dx * dx)) * (in_u[E] - 2.0f * in_u[global_index] + in_u[W]) +
		(1 / (dy * dy)) * (in_u[N] - 2.0f * in_u[global_index] + in_u[S]);

	float laplacian_v = (1 / (dx * dx)) * (in_v[E] - 2.0f * in_v[global_index] + in_v[W]) +
		(1 / (dy * dy)) * (in_v[N] - 2.0f * in_v[global_index] + in_v[S]);

	float adv_h_x = (1.0f / (2.0f * dx)) * (in_h[W] - in_h[E]);

	float adv_h_y = (1.0f / (2.0f * dy)) * (in_h[N] - in_h[S]);

	float adv_u_x = (1.0f / (2.0f * dx)) * (in_u[W] - in_u[E]);

	float adv_u_y = (1.0f / (2.0f * dy)) * (in_u[N] - in_u[S]);

	float adv_v_x = (1.0f / (2.0f * dx)) * (in_v[W] - in_v[E]);

	float adv_v_y = (1.0f / (2.0f * dy)) * (in_v[N] - in_v[S]);


	float du = 0.0f - (g * adv_h_x) - (b * in_u[global_index]) + (nu * laplacian_u) -
		(in_u[global_index] * adv_u_x + in_v[global_index] * adv_u_y);

	float dv = 0.0f - (g * adv_h_y) - (b * in_v[global_index]) + (nu * laplacian_v) -
		(in_u[global_index] * adv_v_x + in_v[global_index] * adv_v_y);

	float dh = 0.0f - (in_u[global_index] * adv_h_x) - (in_h[global_index] * adv_u_x) -
		(in_v[global_index] * adv_h_y) - (in_h[global_index] * adv_v_y) - (mean_H * adv_u_x) - (mean_H * adv_v_y);



	out_u[global_index] = in_u[global_index] + dt*du;

	out_v[global_index] = in_v[global_index] + dt*dv;

	out_h[global_index] = in_h[global_index] + dt*dh;


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

	float* u, * v, * h, * out_u, * out_v, * out_h;

	float* d_u, * d_v, * d_h, * d_out_u, * d_out_v, * d_out_h;

	const char* filename = "shallow_water.dat";

	float amplitude_h = 2.0f;
	float waist_h = 0.01f;

	int T_it = 5000;





	CudaSafeCall(cudaMalloc((void**)&d_u, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_v, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_h, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_out_u, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_out_v, sizeof(float) * Nx * Ny));
	CudaSafeCall(cudaMalloc((void**)&d_out_h, sizeof(float) * Nx * Ny));

	u = (float*)malloc(sizeof(float) * Nx * Ny);
	v = (float*)malloc(sizeof(float) * Nx * Ny);
	h = (float*)malloc(sizeof(float) * Nx * Ny);
	out_u = (float*)malloc(sizeof(float) * Nx * Ny);
	out_v = (float*)malloc(sizeof(float) * Nx * Ny);
	out_h = (float*)malloc(sizeof(float) * Nx * Ny);


	memset(u, 0.0f, Nx * Ny * sizeof(float));
	memset(v, 0.0f, Nx * Ny * sizeof(float));
	memset(h, 0.0f, Nx * Ny * sizeof(float));
	for (int j = 0; j < Ny; j++) {
		for (int i = 0; i < Nx; i++) {

			int glob_ind = i + Nx * j;
			float x0 = (float)i * dx - 0.5f * dx * Nx;
			float y0 = (float)j * dy - 0.5f * dy * Ny;

			h[glob_ind] = mean_H + amplitude_h * exp(-(x0 * x0 + y0 * y0) / (2.0f * waist_h * waist_h));


		}
	}


	CudaSafeCall(cudaMemcpy(d_u, u, sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_v, v, sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_h, h, sizeof(float) * Nx * Ny, cudaMemcpyHostToDevice));

	dim3 BLOCK2D = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 GRID2D = dim3(Nx / BLOCK_SIZE_X, Ny / BLOCK_SIZE_Y, 1);

	vec_add << <GRID2D, BLOCK2D >> > (d_out_u, d_out_v, d_out_h, d_u, d_v, d_h);
	CudaCheckError();
	CudaSafeCall(cudaMemcpy(d_u, d_out_u, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));
	CudaSafeCall(cudaMemcpy(d_v, d_out_v, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));
	CudaSafeCall(cudaMemcpy(d_h, d_out_h, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));



	float totaltime = 0.0f;
	float elapsedTime;

	for (int i = 0; i < T_it; i++) {

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Declare kernel (CUDA function)

		vec_add << <GRID2D, BLOCK2D >> > (d_out_u, d_out_v, d_out_h, d_u, d_v, d_h);
		CudaCheckError();
		CudaSafeCall(cudaMemcpy(d_u, d_out_u, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));
		CudaSafeCall(cudaMemcpy(d_v, d_out_v, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));
		CudaSafeCall(cudaMemcpy(d_h, d_out_h, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToDevice));



		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		totaltime += elapsedTime;
		printf("Time: %f  ms\n", elapsedTime);
	}

	printf("%f \n", (totaltime / T_it));


	CudaSafeCall(cudaMemcpy(out_h, d_out_h, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(out_u, d_out_u, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(out_v, d_out_v, sizeof(float) * Nx * Ny, cudaMemcpyDeviceToHost));

	print2DGnuplot(out_h, filename);

	float max_out_h = out_h[64 + 128*64];
	float max_out_u = out_u[64 + 128*64];
	float max_out_v = out_v[64 + 128*64];

	printf("%f \n", max_out_h);
	printf("%f \n", max_out_u);
	printf("%f \n", max_out_v);

	//print2DGnuplot(out_h, filename);

	cudaFree(d_u); //Free the enslaved memory!!!!
	cudaFree(d_v);
	cudaFree(d_h);
	cudaFree(d_out_u); //Free the enslaved memory!!!!
	cudaFree(d_out_v);
	cudaFree(d_out_h);
	free(u);
	free(v);
	free(h);
	free(out_u);
	free(out_v);
	free(out_h);

	return 0;

}
