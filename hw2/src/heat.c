#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r) * (num) + (c))

#ifndef SIZE
#define SIZE 1000
#endif

void step_kernel_ref(const size_t ni, const size_t nj, const float fact, const float* temp_in, float* temp_out) {
  size_t i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  // loop over all points in domain (except boundary)
  for (size_t j = 1; j < nj - 1; j++) {
    for (size_t i = 1; i < ni - 1; i++) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i - 1, j);
      ip10 = I2D(ni, i + 1, j);
      i0m1 = I2D(ni, i, j - 1);
      i0p1 = I2D(ni, i, j + 1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
      d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
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

  printf("MATRIX SIZE: %zux%zu = %zu B = %.3f GB\n", ni, nj, size, (double)size / 1e9);

  double start = clock();

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);
  temp1 = (float*)malloc(size);
  temp2 = (float*)malloc(size);

  // init with random data
  for (size_t i = 0; i < ni * nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand() / (float)(RAND_MAX / 100.0f);
  }

  // CPU execution
  for (istep = 0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref = temp_tmp;
  }

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for (size_t i = 0; i < ni * nj; ++i) {
    if (abs(temp1[i] - temp1_ref[i]) > maxError) {
      maxError = abs(temp1[i] - temp1_ref[i]);
    }
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

  double end = (clock() - start) / (double)CLOCKS_PER_SEC * 1000.0;  // milliseconds

  printf("\nElapsed time = %.2lf ms = %.2lf s\n", end, end / 1000.0);
  return 0;
}
