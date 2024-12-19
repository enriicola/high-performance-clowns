#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define PI25DT 3.141592653589793238462643

#define INTERVALS 10000000000
#define MASTER_NODE 0
#define NTHREADS_TOTAL 20

int main(int argc, char **argv) {
  long long int i, intervals = INTERVALS;
  double x, dx, f, local_sum, global_sum, pi;
  double time2;

  int NPROCS = 10;
  int NTHREADS = NTHREADS_TOTAL / NPROCS;

  if (argc >= 2)
    intervals = atoll(argv[1]);

  if (argc >= 3) {
    NPROCS = atoi(argv[2]);
    NTHREADS = NTHREADS_TOTAL / NPROCS;
  }

  time_t time1 = clock();

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == MASTER_NODE) {
    printf("Number of intervals: %lld\n", intervals);
    printf("Number of threads: %d\n", NTHREADS);
    printf("Number of procs: %d\n", NPROCS);
  }

  long long int chunk_size = intervals / size;  // How many blocks per node
  long long int start = (rank == 0) ? rank * chunk_size : rank * chunk_size + 1;
  long long int end = (rank == size - 1) ? intervals : start + chunk_size - 1;

  local_sum = 0.0;
  dx = 1.0 / (double)intervals;

#ifdef PARALLEL
#pragma omp parallel for num_threads(NTHREADS) private(x, f) reduction(+ : local_sum)
#endif
  for (i = start; i <= end; i++) {
    x = dx * ((double)(i - 0.5));
    f = 4.0 / (1.0 + x * x);
    local_sum += f;
  }

  // (send_bf, recv_bf, n_elems, datatype_elems, mpi_op, receiver, comm)
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);

  double end_time = MPI_Wtime();

  // Compute and print the result on rank 0
  if (rank == MASTER_NODE) {
    pi = dx * global_sum;
    printf("Computed PI %.24f\n", pi);
    printf("The true PI %.24f\n\n", PI25DT);
    printf("Elapsed time (s) = %.2lf\n", end_time - time1);
  }

  MPI_Finalize();
  return 0;
}
