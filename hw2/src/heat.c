#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // Added for time measurement

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r) * (num) + (c))

// Colab maximum NTHREADS is 2
#ifndef NTHREADS
#define NTHREADS 2
#endif

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE dynamic
#endif

#ifndef SIZE
#define SIZE 1000
#endif

/*
 * `step_kernel_mod` is currently a direct copy of the CPU reference solution
 * `step_kernel_ref` below. Accelerate it to run as a CUDA kernel.
 */
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out) {
#ifdef PARALLEL
#pragma omp parallel for num_threads(NTHREADS) schedule(OMP_SCHEDULE)
#endif
  // loop over all points in domain (except boundary)
  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      // find indices into linear memory
      // for central point and neighbours
      int ij = I2D(ni, i, j);
      int im1j = I2D(ni, i - 1, j);
      int ip1j = I2D(ni, i + 1, j);
      int ijm1 = I2D(ni, i, j - 1);
      int ijp1 = I2D(ni, i, j + 1);

      // evaluate derivatives
      float dx2 = temp_in[im1j] - 2 * temp_in[ij] + temp_in[ip1j];
      float dy2 = temp_in[ijm1] - 2 * temp_in[ij] + temp_in[ijp1];

      // update temperatures
      temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
    }
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out) {
  int ij, im1j, ip1j, ijm1, ijp1;
  float dx2, dy2;

  // loop over all points in domain (except boundary)
  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      // find indices into linear memory
      // for central point and neighbours
      ij = I2D(ni, i, j);
      im1j = I2D(ni, i - 1, j);
      ip1j = I2D(ni, i + 1, j);
      ijm1 = I2D(ni, i, j - 1);
      ijp1 = I2D(ni, i, j + 1);

      // evaluate derivatives
      dx2 = temp_in[im1j] - 2 * temp_in[ij] + temp_in[ip1j];
      dy2 = temp_in[ijm1] - 2 * temp_in[ij] + temp_in[ijp1];

      // update temperatures
      temp_out[ij] = temp_in[ij] + fact * (dx2 + dy2);
    }
  }
}

int main() {
  int istep;
  int nstep = 200;  // number of time steps
  const size_t ni = SIZE;
  const size_t nj = SIZE;
  float tfac = 8.418e-5;  // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const size_t size = ni * nj * sizeof(float);

  printf("Matrix size: %zux%zu (%.3f GB)\n", ni, nj, (double)size / 1e9);

  double time1 = clock();

  // we align in order to vectorize
  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  temp1 = (float*)malloc(size);
  temp2 = (float*)malloc(size);

  // Initialize with random data
  for (int i = 0; i < ni * nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand() / (float)(RAND_MAX / 100.0f);
  }

  // Execute the CPU-only reference version
  for (istep = 0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref = temp_tmp;
  }

  // Execute the modified version using same data
  for (istep = 0; istep < nstep; istep++) {
    step_kernel_mod(ni, nj, tfac, temp1, temp2);

    // swap the temperature pointers
    temp_tmp = temp1;
    temp1 = temp2;
    temp2 = temp_tmp;
  }

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for (int i = 0; i < ni * nj; ++i) {
    if (abs(temp1[i] - temp1_ref[i]) > maxError) {
      maxError = abs(temp1[i] - temp1_ref[i]);
    }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  double time2 = (clock() - time1) / (double)CLOCKS_PER_SEC * 1000.0;

  printf("Elapsed time: %f ms\n", time2);

  free(temp1_ref);
  free(temp2_ref);
  free(temp1);
  free(temp2);

  return 0;
}
