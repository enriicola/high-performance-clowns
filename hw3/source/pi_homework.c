#include <omp.h>
#include <stdio.h>
#include <time.h>

#define PI25DT 3.141592653589793238462643

#define INTERVALS 100000000000

int main(int argc, char **argv) {
  long int i, intervals = INTERVALS;
  double x, dx, f, sum, pi;
  double time2;

<<<<<<< HEAD
#ifdef PARALLEL
  double time1 = omp_get_wtime();
#else
  time_t time1 = clock();
#endif

  printf("Number of intervals: %ld\n", intervals);
=======
  time_t time1 = clock();
>>>>>>> ec11d75fb1395fc9c3ee76fecebe509e9f885a55

  sum = 0.0;
  dx = 1.0 / (double)intervals;

#ifdef PARALLEL
  omp_set_num_threads(20);
<<<<<<< HEAD
#pragma omp parallel for private(x, f) reduction(+ : sum)
=======
#pragma omp parallel for private(x, f) reduction(+ : local_sum)
>>>>>>> ec11d75fb1395fc9c3ee76fecebe509e9f885a55
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
