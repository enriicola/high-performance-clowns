#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#define PI25DT 3.141592653589793238462643

#define INTERVALS 10000000000
#define MASTER_NODE 0

int main(int argc, char **argv) {
  long int i, intervals = INTERVALS;
  double x, dx, f, local_sum, global_sum, pi;
  double time2;

  time_t time1 = clock();

  int size, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0)
    printf("Number of intervals: %ld\n", intervals);

  printf("Node %d: working...\n", rank);

  long int chunk_size = intervals / size;  // How many blocks per node
  long int start = rank * chunk_size + 1;
  long int end = (rank == size - 1) ? intervals : start + chunk_size - 1;

  local_sum = 0.0;
  dx = 1.0 / (double)intervals;

#ifdef PARALLEL
  omp_set_num_threads(20);
#pragma omp parallel for private(x, f) reduction(+ : local_sum)
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
  if (rank == 0) {
    pi = dx * global_sum;
    printf("Computed PI %.24f\n", pi);
    printf("The true PI %.24f\n\n", PI25DT);
    printf("Elapsed time (s) = %.2lf\n", end_time - time1);
  }

  MPI_Finalize();
  return 0;
}
