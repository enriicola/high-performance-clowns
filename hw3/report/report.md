# Assignment 3: MPI

The goal is to distribute the program between the different nodes (in out case different 
processes on the same machine). 

# Hardware

For this first assignment, we executed the C code using the Software 2 (SW2) workstations, with the following characteristics.

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="./images/hardware.png" alt="Distributed" width="70%" />
  </figure>
</div>

## Algorithm analysis

Given the function

$$f(x) = \frac{4}{1 + x^2}$$

It's proven that the integral in $[0, 1]$ is equal to $\pi$:

$$\int_0^1 f(x)\ dx = \pi$$

Through **midpoint Riemann sums**, this integral can be approximated by:

$$\pi \approx \frac{1}{n} \sum_{i=1}^n \frac{4}{1 + (\frac{i-0.5}{n})^2}$$

$(i - 0.5)/n$ is indeed the midpoint of the $i$-th subinterval. Since it discretizes the integral, it becomes an approximation of $\pi$.

### Parallelization strategy

The sequential case is the following:

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="./images/sequential.png" alt="Sequential" width="70%" />
    <figcaption>Figure 1: sequential case</figcaption>
  </figure>
</div>

As we can see, the sum is performed alltogether on the same node. The simplest yet most powerful way to parallelize this sum is to split the computations on different nodes, making them calculate only a chunk of the total sum.

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="./images/mpi.png" alt="Distributed" width="80%" />
    <figcaption>Figure 2: parallel case</figcaption>
  </figure>
</div>

The final step would be re-aggregate all the partial sums into the same global result, i.e. **reduce** them.

### Vectorization

Now that the program is distributed, we can start thinking at the other optimization aspects. The first thing to tune is the compiler, in our case we used `mpiicx` with all the optimization flag needed. The command ran to compile the program is

```bash
mpiicx -g -O3 -xHost -qopenmp -qopt-report=3 -ffast-math pi_homework.c
```

while the one to execute it is

```bash
mpirun -np 10 ./pi_homework
```

### Parallelization

Another way to improve the performances is to use multithreading. We used **OpenMP** to parallelize the original code in this way:

```c
#pragma omp parallel for num_threads(NTHREADS) private(x, f) reduction(+ : sum)
  for (i = 1; i <= intervals; i++) {
    x = dx * ((double)(i - 0.5));
    f = 4.0 / (1.0 + x * x);
    sum = sum + f;
  }
```

That is, simply subdivide the for loop on different threads and apply the reduction on the sum.

### MPI workload distribution

Now that we know how the algorithm works, we can decide how to divide the workload in each MPI node. We start by noticing that it is a sum over $n$ elements (in the code $n =$ `INTERVALS`), so, if we have $m$ worker nodes with same resources and performances, we can divide this sum into $m/\texttt{INTERVALS}$ chunks

```c
int size, rank;
MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );

long int chunk_size = intervals / size;
```

and, for each node, determine the starting index and the ending one:

```c
long int start = rank * chunk_size + 1;
long int end = (rank == size - 1) ? intervals : start + chunk_size - 1;
```

at this point, the local sum on the node is computed

```c
double local_sum = 0.0;
double x, f;
double dx = 1 / (double)intervals; // 1 / n

for (long int i = start; i <= end; i++) {
  x = dx * ((double)(i - 0.5));
  f = 4.0 / (1.0 + x * x);
  local_sum += f;
}
```

At the end of the loop, the local sum on the node will be computed. After this loop, we reduce all the results by summing the partial sums on the master node ($0$ in our case), getting the final result.

```c
double global_sum;
// (send_bf, recv_bf, n_elems, datatype_elems, mpi_op, receiver, comm)
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MASTER_NODE, MPI_COMM_WORLD);

if(rank == MASTER_NODE){
  double pi = dx * global_sum;
  printf("Computed PI %.24f\n", pi);
  printf("The true PI %.24f\n\n", PI25DT);
}
```

## Performance evaluation

In this program, the only heavy hotspot is `loop in main at pi_homework.c:26`, that is the one that computes the local sums. The following measurements are taken considering the global execution time and the hotspot execution time (since it's only one).

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="./images/comparison.png" alt="Distributed" width="80%" />
  </figure>
</div>
<div style="text-align:center;"> execution time comparison</div>

#### Parallel execution time

|    INTERVALS    | GFLOPS | Time (s) | Hotspot Time (s) |
| :-------------: | :----: | :------: | :--------------: |
| 100.000.000.000 | 75.92  |   7.90   |       6.25       |
| 300.000.000.000 | 78.45  |  22.94   |      18.74       |
| 500.000.000.000 | 77.91  |  38.51   |      33.68       |
| 700.000.000.000 | 75.36  |  55.73   |      48.08       |

#### MPI execution time

|    INTERVALS    | Time (s) |
| :-------------: | :------: |
| 100.000.000.000 |  11.85   |
| 300.000.000.000 |  37.96   |
| 500.000.000.000 |  66.49   |
| 700.000.000.000 |  95.72   |


In general, with multithreading we avoid introducing useless overheads due to process creation

<div style="display: flex; justify-content: center; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="./images/speedup.png" alt="Distributed" width="80%" />
  </figure>
</div>

<div style="text-align:center;">Speedup (parallel vs. MPI)</div>

#### Speedup table

|  INTERVALS   | Speedup |
| :----------: | :-----: |
| 100000000000 |  1.50   |
| 300000000000 |  1.66   |
| 500000000000 |  1.73   |
| 700000000000 |  1.72   |

Because of this, we see that the parallel program is $1.6$ times faster than the MPI program.

# Conclusions

The main concept to reason on in this case is the scalability. Multithreading is useful and fast, but, in the best case, we can have no more than $\approx 64$ threads in a single machine, while we can add potentially infinite nodes to an MPI cluster.