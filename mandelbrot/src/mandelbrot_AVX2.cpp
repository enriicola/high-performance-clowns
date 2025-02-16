#include <immintrin.h>

#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>

// OMP
#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE guided
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

int main(int argc, char **argv) {
  const int totalPixels = HEIGHT * WIDTH;
  int *const image = new int[totalPixels];
  size_t bytes = totalPixels * sizeof(int);

  std::cout << "Image size: "
            << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0))
            << " GB" << std::endl;

  const auto start = chrono::steady_clock::now();

#ifdef PARALLEL
#pragma omp parallel for num_threads(NUM_THREADS) schedule(OMP_SCHEDULE)
#endif
  // 4 pixels in parallel
  for (int pos = 0; pos < totalPixels; pos += 4) {
    // z = 0
    __m256d z_real = _mm256_setzero_pd();
    __m256d z_imag = _mm256_setzero_pd();

    // c for 4 pixels
    __m256d c_real = _mm256_set_pd(
        ((pos + 3) % WIDTH) * STEP + MIN_X,
        ((pos + 2) % WIDTH) * STEP + MIN_X,
        ((pos + 1) % WIDTH) * STEP + MIN_X,
        (pos % WIDTH) * STEP + MIN_X);
    __m256d c_imag = _mm256_set_pd(
        ((pos + 3) / WIDTH) * STEP + MIN_Y,
        ((pos + 2) / WIDTH) * STEP + MIN_Y,
        ((pos + 1) / WIDTH) * STEP + MIN_Y,
        (pos / WIDTH) * STEP + MIN_Y);

    // active mask: each lane is active (all bits 1)
    __m256d active = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
    for (int i = 1; i <= ITERATIONS; i++) {
      // z^2
      __m256d z_real_sq = _mm256_mul_pd(z_real, z_real);
      __m256d z_imag_sq = _mm256_mul_pd(z_imag, z_imag);
      // z_real^2 - z_imag^2 + c_real
      __m256d new_z_real = _mm256_add_pd(_mm256_sub_pd(z_real_sq, z_imag_sq), c_real);
      // 2*z_real*z_imag + c_imag
      __m256d tmp = _mm256_mul_pd(z_real, z_imag);
      __m256d new_z_imag = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), tmp), c_imag);

      // update active lanes
      z_real = _mm256_blendv_pd(z_real, new_z_real, active);
      z_imag = _mm256_blendv_pd(z_imag, new_z_imag, active);

      // |z|^2 = z_real^2 + z_imag^2
      __m256d mag_sq = _mm256_add_pd(_mm256_mul_pd(z_real, z_real),
                                     _mm256_mul_pd(z_imag, z_imag));
      // |z|^2 >= 4.
      __m256d diverged_mask = _mm256_cmp_pd(mag_sq, _mm256_set1_pd(4.0), _CMP_GE_OQ);
      int diverged = _mm256_movemask_pd(diverged_mask);
      if (diverged) {  // if any of the 4 pixel diverged
        for (int j = 0; j < 4; ++j) {
          if (diverged & (1 << j)) {  // if the j-th pixel diverged
            image[pos + j] = i;
          }
        }
        // remove diverged lanes from future iterations
        active = _mm256_andnot_pd(diverged_mask, active);
        if (_mm256_movemask_pd(active) == 0)
          break;
      }
    }

    // for pixels that never diverged
    int still_active = _mm256_movemask_pd(active);
    if (still_active) {
      for (int j = 0; j < 4; ++j) {
        if (still_active & (1 << j)) {
          image[pos + j] = 0;
        }
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