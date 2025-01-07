#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r) * (num) + (c))

#ifndef BLOCKDIM_X
#define BLOCKDIM_X 16
#endif

#ifndef BLOCKDIM_Y
#define BLOCKDIM_Y 16
#endif

#ifndef SIZE
#define SIZE 1000
#endif

#ifndef NSTEPS
#define NSTEPS 200
#endif

/**
 * @brief Performs a step of the heat equation simulation (CUDA).
 *
 * This function updates the temperature distribution by computing the
 * second derivatives in both the x and y directions and applying the
 * scaling factor to calculate the new temperature values.
 *
 * @param ni      Number of grid points in the i (x) direction.
 * @param nj      Number of grid points in the j (y) direction.
 * @param fact    Scaling factor applied to the computed derivatives.
 * @param temp_in Pointer to the input temperature array.
 * @param temp_out Pointer to the output temperature array where results are stored.
 *
 * @global The kernel is executed by multiple thread blocks in a 2D grid
 */
__global__ void step_kernel_mod_dev(const size_t ni, const size_t nj, const float fact, const float* temp_in, float* temp_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if ((i > 0 && i < ni - 1) && (j > 0 && j < nj - 1)) {
    /*
      im1j = (i-1)j
      ip1j = (i+1)j
      ijm1 = i(j-1)
      ijp1 = i(j+1)
    */

    // find indices into linear memory
    // for central point and neighbours
    size_t ij = I2D(ni, i, j);        // T[i,j]
    size_t im1j = I2D(ni, i - 1, j);  // T[i-1,j]
    size_t ip1j = I2D(ni, i + 1, j);  // T[i+1,j]
    size_t ijm1 = I2D(ni, i, j - 1);  // T[i,j-1]
    size_t ijp1 = I2D(ni, i, j + 1);  // T[i,j+1]

    // evaluate derivatives
    // we suppose delta x^2 to be 1?
    float dx2 = temp_in[ip1j] - 2.0f * temp_in[ij] + temp_in[im1j];
    float dy2 = temp_in[ijp1] - 2.0f * temp_in[ij] + temp_in[ijm1];

    // update temperatures
    temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
  }
}

/**
 * @brief Performs a step of the heat equation simulation (sequential).
 *
 * This function updates the temperature distribution by computing the
 * second derivatives in both the x and y directions and applying the
 * scaling factor to calculate the new temperature values.
 *
 * @param ni      Number of grid points in the i (x) direction.
 * @param nj      Number of grid points in the j (y) direction.
 * @param fact    Scaling factor applied to the computed derivatives.
 * @param temp_in Pointer to the input temperature array.
 * @param temp_out Pointer to the output temperature array where results are stored.
 */
__host__ void step_kernel_ref(const size_t ni, const size_t nj, const float fact, float* temp_in, float* temp_out) {
  // loop over all points in domain (except boundary)
  for (size_t i = 1; i < ni - 1; i++) {
    for (size_t j = 1; j < nj - 1; j++) {
      /*
        im1j = (i-1)j
        ip1j = (i+1)j
        ijm1 = i(j-1)
        ijp1 = i(j+1)
      */

      // find indices into linear memory
      // for central point and neighbours
      size_t ij = I2D(ni, i, j);        // T[i,j]
      size_t im1j = I2D(ni, i - 1, j);  // T[i-1,j]
      size_t ip1j = I2D(ni, i + 1, j);  // T[i+1,j]
      size_t ijm1 = I2D(ni, i, j - 1);  // T[i,j-1]
      size_t ijp1 = I2D(ni, i, j + 1);  // T[i,j+1]

      // evaluate derivatives
      // we suppose delta x^2 to be 1?
      float dx2 = temp_in[ip1j] - 2.0f * temp_in[ij] + temp_in[im1j];
      float dy2 = temp_in[ijp1] - 2.0f * temp_in[ij] + temp_in[ijm1];

      // update temperatures
      temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
    }
  }
}

// Launches the CUDA kernel
__host__ void step_kernel_mod(const size_t ni, const size_t nj, const float fact, const float* temp_in_d, float* temp_out_d) {
  dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  /*
    we divide n_rows/block.x and n_cols/block.y. To avoid checking
    if the n_rows and n_cols are divisible by block.x and block.y,
    we add block.x - 1 and block.y - 1 at the numerator, so we don't
    miss the last cell.
  */
  dim3 numBlocks((ni + threadsPerBlock.x - 1) / threadsPerBlock.x, (nj + threadsPerBlock.y - 1) / threadsPerBlock.y);
  step_kernel_mod_dev<<<numBlocks, threadsPerBlock>>>(ni, nj, fact, temp_in_d, temp_out_d);
}

int main() {
  int nstep = NSTEPS;                           // n_iterations
  const size_t ni = SIZE;                       // rows
  const size_t nj = SIZE;                       // cols
  float tfac = 8.418e-5f;                       // thermal diffusivity of silver
  const size_t size = ni * nj * sizeof(float);  // matrix size

  printf("BLOCKDIM: %dx%d\n", BLOCKDIM_X, BLOCKDIM_Y);
  printf("MATRIX SIZE: %zux%zu = %zu B = %.3f GB\n", ni, nj, size, (double)size / 1e9);

  double start = clock();

  // Host allocations
  // temp1_ref: true values (input)
  // temp2_ref: true values (output)
  // temp*: initial values to copy to device
  float* temp1_ref = (float*)malloc(size);
  float* temp2_ref = (float*)malloc(size);
  float* temp1 = (float*)malloc(size);
  float* temp2 = (float*)malloc(size);

  // Random init
  for (size_t i = 0; i < ni * nj; ++i) {
    float v = (float)rand() / ((float)RAND_MAX / 100.0f);
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = v;
  }

  // CPU reference
  for (int step = 0; step < nstep; step++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
    // the output becomes the next step's input
    float* tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref = tmp;
  }

  /* CUDA */

  // Device allocations
  // allocate and copy the initial values to device
  float *temp1_d, *temp2_d;

  cudaMalloc(&temp1_d, size);
  cudaMalloc(&temp2_d, size);
  cudaMemcpy(temp1_d, temp1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(temp2_d, temp2, size, cudaMemcpyHostToDevice);

  // GPU steps
  for (int step = 0; step < nstep; step++) {
    step_kernel_mod(ni, nj, tfac, temp1_d, temp2_d);
    // the output becomes the next step's input
    float* tmp = temp1_d;
    temp1_d = temp2_d;
    temp2_d = tmp;
  }

  // Copy back
  // temp1 is the final output after the last swap
  cudaMemcpy(temp1, temp1_d, size, cudaMemcpyDeviceToHost);

  // Check error
  float maxError = 0.0f;
  for (size_t i = 0; i < ni * nj; ++i) {
    float diff = fabsf(temp1[i] - temp1_ref[i]);
    if (diff > maxError) maxError = diff;
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  free(temp1_ref);
  free(temp2_ref);
  free(temp1);
  free(temp2);
  cudaFree(temp1_d);
  cudaFree(temp2_d);

  double end = (clock() - start) / (double)CLOCKS_PER_SEC * 1000.0;  // milliseconds

  printf("\nElapsed time = %.2lf ms = %.2lf s\n", end, end / 1000.0);
  return 0;
}
