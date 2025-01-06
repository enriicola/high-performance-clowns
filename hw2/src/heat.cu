#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define I2D(num, c, r) ((r) * (num) + (c))

__global__ void step_kernel_mod_dev(int ni, int nj, float fact, const float* temp_in, float* temp_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > 0 && i < ni - 1 && j > 0 && j < nj - 1) {
    int i00 = I2D(ni, i, j);
    int im10 = I2D(ni, i - 1, j);
    int ip10 = I2D(ni, i + 1, j);
    int i0m1 = I2D(ni, i, j - 1);
    int i0p1 = I2D(ni, i, j + 1);
    float d2tdx2 = temp_in[im10] - 2.0f * temp_in[i00] + temp_in[ip10];
    float d2tdy2 = temp_in[i0m1] - 2.0f * temp_in[i00] + temp_in[i0p1];
    temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out) {
  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      int i00 = I2D(ni, i, j);
      int im10 = I2D(ni, i - 1, j);
      int ip10 = I2D(ni, i + 1, j);
      int i0m1 = I2D(ni, i, j - 1);
      int i0p1 = I2D(ni, i, j + 1);
      float d2tdx2 = temp_in[im10] - 2.f * temp_in[i00] + temp_in[ip10];
      float d2tdy2 = temp_in[i0m1] - 2.f * temp_in[i00] + temp_in[i0p1];
      temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
  }
}

// Launches the CUDA kernel
void step_kernel_mod(int ni, int nj, float fact, float* temp_in_d, float* temp_out_d) {
  dim3 block(16, 16);
  dim3 grid((ni + block.x - 1) / block.x, (nj + block.y - 1) / block.y);
  step_kernel_mod_dev<<<grid, block>>>(ni, nj, fact, temp_in_d, temp_out_d);
}

int main() {
  int nstep = 200;
  const int ni = 1000, nj = 1000;
  float tfac = 8.418e-5;
  int size = ni * nj * sizeof(float);

  // Host allocations
  float* temp1_ref = (float*)malloc(size);
  float* temp2_ref = (float*)malloc(size);
  float* temp1 = (float*)malloc(size);
  float* temp2 = (float*)malloc(size);

  // Random init
  for (int i = 0; i < ni * nj; ++i) {
    float v = (float)rand() / (float)(RAND_MAX / 100.0f);
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = v;
  }

  // CPU reference
  for (int step = 0; step < nstep; step++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
    float* tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref = tmp;
  }

  // Device allocations
  float *temp1_d, *temp2_d;
  cudaMalloc(&temp1_d, size);
  cudaMalloc(&temp2_d, size);
  cudaMemcpy(temp1_d, temp1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(temp2_d, temp2, size, cudaMemcpyHostToDevice);

  // GPU steps
  for (int step = 0; step < nstep; step++) {
    step_kernel_mod(ni, nj, tfac, temp1_d, temp2_d);
    float* tmp = temp1_d;
    temp1_d = temp2_d;
    temp2_d = tmp;
  }

  // Copy back
  cudaMemcpy(temp1, temp1_d, size, cudaMemcpyDeviceToHost);

  // Check error
  float maxError = 0;
  for (int i = 0; i < ni * nj; ++i) {
    float diff = fabs(temp1[i] - temp1_ref[i]);
    if (diff > maxError) maxError = diff;
  }

  if (maxError > 0.0005f)
    printf("Problem! Max Error = %.5f\n", maxError);
  else
    printf("Max Error = %.5f (OK)\n", maxError);

  free(temp1_ref);
  free(temp2_ref);
  free(temp1);
  free(temp2);
  cudaFree(temp1_d);
  cudaFree(temp2_d);
  return 0;
}
