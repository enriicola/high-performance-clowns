/******************************************************************************
 * heat_fixed_compare.cu
 *
 * This version runs:
 *   1) CPU 200 steps from the random initial data
 *   2) GPU 200 steps from that SAME random initial data
 * and then compares final states fairly.
 ******************************************************************************/
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r) * (num) + (c))

#ifndef BLOCKDIM_X
#define BLOCKDIM_X 32
#endif

#ifndef BLOCKDIM_Y
#define BLOCKDIM_Y 8
#endif

#ifndef SIZE
#define SIZE 1000
#endif

/******************************************************************************
 * Error-checking helper
 ******************************************************************************/
void handle_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "GPU error: %s\n", cudaGetErrorString(err));
    exit(err);
  }
}

/******************************************************************************
 * GPU kernel
 ******************************************************************************/
__global__ void step_kernel_mod_dev(const size_t ni, const size_t nj,
                                    const float fact,
                                    const float* temp_in,
                                    float* temp_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i > 0 && i < ni - 1) && (j > 0 && j < nj - 1)) {
    // Indices
    size_t ij = I2D(ni, i, j);
    size_t im1j = I2D(ni, i - 1, j);
    size_t ip1j = I2D(ni, i + 1, j);
    size_t ijm1 = I2D(ni, i, j - 1);
    size_t ijp1 = I2D(ni, i, j + 1);

    // second derivatives
    float dx2 = temp_in[ip1j] - 2.0f * temp_in[ij] + temp_in[im1j];
    float dy2 = temp_in[ijp1] - 2.0f * temp_in[ij] + temp_in[ijm1];

    // update
    temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
  }
}

/******************************************************************************
 * Reference CPU version
 ******************************************************************************/
__host__ void step_kernel_ref(const size_t ni, const size_t nj,
                              const float fact,
                              float* temp_in,
                              float* temp_out) {
  for (size_t j = 1; j < nj - 1; j++) {
    for (size_t i = 1; i < ni - 1; i++) {
      size_t ij = I2D(ni, i, j);
      size_t im1j = I2D(ni, i - 1, j);
      size_t ip1j = I2D(ni, i + 1, j);
      size_t ijm1 = I2D(ni, i, j - 1);
      size_t ijp1 = I2D(ni, i, j + 1);

      float dx2 = (temp_in[ip1j] - 2.0f * temp_in[ij] + temp_in[im1j]);
      float dy2 = (temp_in[ijp1] - 2.0f * temp_in[ij] + temp_in[ijm1]);

      temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
    }
  }
}

int main() {
  int nstep = 200;
  const size_t ni = SIZE;  // cols
  const size_t nj = SIZE;  // rows
  float tfac = 8.418e-5f;  // thermal diffusivity of silver
  const size_t size = ni * nj * sizeof(float);

  printf("BLOCKDIM: %dx%d\n", BLOCKDIM_X, BLOCKDIM_Y);
  printf("Matrix size: %zux%zu (%.3f GB)\n", ni, nj, (double)size / 1e9);

  /*************************
   * Host allocations
   *************************/
  float* init_data = (float*)malloc(size);
  float* temp1_ref = (float*)malloc(size);
  float* temp2_ref = (float*)malloc(size);
  float* gpu_res = (float*)malloc(size);

  /*************************
   * Device allocations
   *************************/
  float *temp1_d, *temp2_d;
  handle_error(cudaMalloc((void**)&temp1_d, size));
  handle_error(cudaMalloc((void**)&temp2_d, size));

  // init with random data
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < ni * nj; ++i) {
    float v = (float)rand() / (float)(RAND_MAX / 100.0f);
    temp1_ref[i] = temp2_ref[i] = init_data[i] = v;
  }

  /****************************************************************
   * CPU Simulation from init_data
   ****************************************************************/

  // time the CPU
  double cpuStart = clock();

  for (int step = 0; step < nstep; step++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    float* tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref = tmp;
  }

  double cpuEnd = clock();
  double cpuTimeMs = (cpuEnd - cpuStart) / (double)CLOCKS_PER_SEC * 1000.0;
  printf("\n--- CPU simulation ---\n");
  printf("CPU time: %.2f ms (%.2f s)\n", cpuTimeMs, cpuTimeMs / 1000.0);

  // after the final swap, the CPU results are in temp1_ref

  /****************************************************************
   * GPU Simulation from SAME init_data
   ****************************************************************/
  // copy init_data to GPU arrays
  handle_error(cudaMemcpy(temp1_d, init_data, size, cudaMemcpyHostToDevice));
  handle_error(cudaMemcpy(temp2_d, init_data, size, cudaMemcpyHostToDevice));

  // time measurement
  cudaEvent_t start, stop;
  handle_error(cudaEventCreate(&start));
  handle_error(cudaEventCreate(&stop));

  handle_error(cudaEventRecord(start, 0));

  dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 numBlocks((ni + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (nj + threadsPerBlock.y - 1) / threadsPerBlock.y);

  for (int step = 0; step < nstep; step++) {
    step_kernel_mod_dev<<<numBlocks, threadsPerBlock>>>(ni, nj, tfac, temp1_d, temp2_d);

    float* tmp = temp1_d;
    temp1_d = temp2_d;
    temp2_d = tmp;
  }

  handle_error(cudaEventRecord(stop, 0));
  handle_error(cudaEventSynchronize(stop));

  float gpuTimeMs = 0.0f;
  handle_error(cudaEventElapsedTime(&gpuTimeMs, start, stop));

  printf("\n--- GPU simulation ---\n");
  printf("GPU time: %.2f ms (%.2f s)\n", gpuTimeMs, gpuTimeMs / 1000.0);

  handle_error(cudaEventDestroy(start));
  handle_error(cudaEventDestroy(stop));

  /****************************************************************
   * GPU vs CPU Results
   ****************************************************************/
  // temp1_d contains the final GPU result after the last swap
  handle_error(cudaMemcpy(gpu_res, temp1_d, size, cudaMemcpyDeviceToHost));

  float maxError = 0.0f;
  for (size_t i = 0; i < ni * nj; ++i) {
    float diff = fabsf(gpu_res[i] - temp1_ref[i]);
    if (diff > maxError) maxError = diff;
  }

  if (maxError > 0.0005f)
    printf("\nProblem! Max Error of %.5f is NOT within acceptable bounds.\n",
           maxError);
  else
    printf("\nSuccess! Max Error of %.5f is within acceptable bounds.\n",
           maxError);

  free(init_data);
  free(temp1_ref);
  free(temp2_ref);
  free(gpu_res);
  handle_error(cudaFree(temp1_d));
  handle_error(cudaFree(temp2_d));

  return 0;
}
