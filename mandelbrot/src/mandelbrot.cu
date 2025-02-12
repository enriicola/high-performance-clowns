#include <cuda.h>

#include <chrono>
#include <fstream>
#include <iostream>

#ifndef BLOCKDIM_X
#define BLOCKDIM_X 32
#endif

#ifndef BLOCKIM_Y
#define BLOCKDIM_Y 32
#endif

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#ifndef RESOLUTION
#define RESOLUTION 1000
#endif

#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2  // Degree of the polynomial

#define ITERATIONS 1000  // Maximum number of iterations

using namespace std;

// since the STD namespace is not usable on the device
__device__ struct complex {
  double re;
  double im;

  // for z = a + bi, |z|^2 = a^2 + b^2.
  __device__ double abs2() const {
    return re * re + im * im;
  }

  // for z = a + bi, z^2 = (a^2 - b^2) + 2abi
  __device__ complex square() {
    return {re * re - im * im, 2.0 * re * im};
  }
};

void handle_error(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "GPU error: %s\n", cudaGetErrorString(err));
    exit(err);
  }
}

__global__ void mandelbrot_kernel(int *const image) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= WIDTH || row >= HEIGHT)
    return;

  int pos = row * WIDTH + col;

  image[pos] = 0;

  complex c = {col * STEP + MIN_X, row * STEP + MIN_Y};
  complex z = {0.0, 0.0};
  for (int i = 1; i <= ITERATIONS; i++) {
    // z = z^2
    z = z.square();

    // z^2 = z^2 + c;
    z.re += c.re;
    z.im += c.im;

    // If it is convergent
    if (z.abs2() >= 4.0) {  // if abs > 2.0, then abs^2 > 4.0
      image[pos] = i;
      return;
    }
  }
}

int main(int argc, char **argv) {
  const int totalPixels = HEIGHT * WIDTH;
  size_t bytes = totalPixels * sizeof(int);

  // Host allocations
  int *const h_image = new int[totalPixels];

  // Device allocations
  int *d_image = nullptr;
  handle_error(cudaMalloc((void **)&d_image, bytes));

  cout << "Image size: "
       << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0))
       << " GB" << endl;

  const auto start = chrono::steady_clock::now();

  dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 numBlocks((HEIGHT + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (WIDTH + threadsPerBlock.y - 1) / threadsPerBlock.y);

  mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_image);
  handle_error(cudaDeviceSynchronize());

  const auto end = chrono::steady_clock::now();

  // copy result back on the host
  handle_error(cudaMemcpy(h_image, d_image, bytes, cudaMemcpyDeviceToHost));

  cout << "Time elapsed: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " milliseconds." << endl;

  if (argc < 2) {
    cerr << "Please specify the output file as a parameter." << endl;
    handle_error(cudaFree(d_image));
    delete[] h_image;
    return -1;
  }

  cout << "Writing to file..." << endl;
  ofstream matrix_out(argv[1], ios::trunc);
  if (!matrix_out) {
    cout << "Unable to open file." << endl;
    handle_error(cudaFree(d_image));
    delete[] h_image;
    return -2;
  }

  for (int row = 0; row < HEIGHT; row++) {
    for (int col = 0; col < WIDTH; col++) {
      matrix_out << h_image[row * WIDTH + col];
      if (col < WIDTH - 1)
        matrix_out << ',';
    }
    if (row < HEIGHT - 1)
      matrix_out << endl;
  }
  cout << "Done." << endl;

  handle_error(cudaFree(d_image));
  delete[] h_image;
  return 0;
}
