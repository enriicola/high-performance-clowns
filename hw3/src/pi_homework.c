#include <omp.h>
#include <stdio.h>
#include <time.h>

#define PI25DT 3.141592653589793238462643

#ifndef INTERVALS
#define INTERVALS 100000000000
#endif

#define NTHREADS 20

int main(int argc, char **argv) {
  long long int i, intervals = INTERVALS;
  double x, dx, f, sum, pi;
  double time1, time2;

#ifdef PARALLEL
  time1 = omp_get_wtime();
#else
  time1 = clock();
#endif

  sum = 0.0;
  dx = 1.0 / (double)intervals;

#ifdef PARALLEL
#pragma omp parallel for num_threads(NTHREADS) private(x, f) reduction(+ : sum)
#endif
  for (i = 1; i <= intervals; i++) {
    x = dx * ((double)(i - 0.5));
    f = 4.0 / (1.0 + x * x);
    sum = sum + f;
  }

  pi = dx * sum;

#ifdef PARALLEL
  time2 = omp_get_wtime() - time1;
#else
  time2 = (clock() - time1) / (double)CLOCKS_PER_SEC;
#endif

  printf("Computed PI %.24f\n", pi);
  printf("The true PI %.24f\n\n", PI25DT);
  printf("Elapsed time (s) = %.2lf\n", time2);

  return 0;
}
