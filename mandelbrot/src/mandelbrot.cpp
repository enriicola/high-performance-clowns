#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>

// OMP
#define NUM_THREADS 20

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

int main(int argc, char **argv) {
  const int totalPixels = HEIGHT * WIDTH;
  int *const image = new int[totalPixels];
  size_t bytes = totalPixels * sizeof(int);

  std::cout << "Image size: "
            << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0))
            << " GB" << std::endl;

  const auto start = chrono::steady_clock::now();

  complex<double> z(0, 0);
#ifdef PARALLEL
  #pragma omp parallel for num_threads(NUM_THREADS) private(z) schedule(dynamic)
#endif
  for (int pos = 0; pos < HEIGHT * WIDTH; pos++) {
    image[pos] = 0;

    const int row = pos / WIDTH;
    const int col = pos % WIDTH;
    const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

    // z = z^2 + c
    z = complex<double>(0, 0);
    for (int i = 1; i <= ITERATIONS; i++) {
      // RaW DEPENDENCY:
      // z[n] = z[n-1]^2 + c
      z = z * z + c;

      // If it diverges
      if (abs(z) >= 2) {
        image[pos] = i;
        break;
      }
    }
  }
  const auto end = chrono::steady_clock::now();

  cout << "Time elapsed: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " milliseconds." << endl;

  cout << "Writing to file..." << endl;
  // Write the result to a file
  if (argc < 2) {
    cout << "Please specify the output file as a parameter." << endl;
    return -1;
  }

  ofstream matrix_out(argv[1], ios::trunc);
  if (!matrix_out) {
    cout << "Unable to open file." << endl;
    return -2;
  }

  for (int row = 0; row < HEIGHT; row++) {
    for (int col = 0; col < WIDTH; col++) {
      matrix_out << image[row * WIDTH + col];
      if (col < WIDTH - 1)
        matrix_out << ',';
    }
    if (row < HEIGHT - 1)
      matrix_out << endl;
  }

  cout << "Done." << endl;
  delete[] image;  // It's here for coding style, but useless
  return 0;
}