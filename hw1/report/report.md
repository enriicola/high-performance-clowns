# HPC - OpenMP report 🏎️ 💻
### Leonardo Gonfiantini,  Christian Parodi, Enrico Pezzano
### December 2024

# Introduction 🎬

The goal of this laboratory is to optimize the implementation of the Discrete Fourier Transform (DFT) algorithm by leveraging OpenMP HPC techniques, with a focus on parallelization and vectorization, especially on the hotspots.

bozza{
    Specifically, the optimization process employs OpenMP to improve performance and scalability. The primary objectives include reducing execution time, maintaining computational accuracy, and addressing challenges such as hotspot identification, vectorization issues, and thread scalability. Through this approach, various strategies—including hotspot analysis, vectorization, and parallelization are explored and evaluated to enhance the efficiency of the DFT implementation.
}

# Algorithm analysis \<emoji del tizio che scrive al pc>

The Fourier Transform is a mathematical transformation used to convert a function of time (or space) into a function of frequency. It is defined as:

$$
\mathcal{F}\{f(t)\} = F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i \omega t} \, dt
$$

For discrete signals, the Discrete Fourier Transform (DFT) is used, which is defined as:

$$
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i \frac{2\pi}{N} k n} = \sum_{n=0}^{N-1} x_n \cdot \left( \cos\left(\frac{2\pi}{N} k n\right) + i \sin\left(\frac{2\pi}{N} k n\right) \right)
$$

where:
- \( X_k \) is the DFT of the sequence \( x_n \)
- \( N \) is the number of points in the sequence
- \( k \) is the index of the output frequency component
- \( n \) is the index of the input time-domain sequence
- \( i \) is the imaginary unit

The inverse DFT (IDFT) is given by:

$$
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{i \frac{2\pi}{N} k n} =\frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot \left( \cos\left(\frac{2\pi}{N} k n\right) + i \sin\left(\frac{2\pi}{N} k n\right) \right)
$$

Notice it's just the DFT of the DFT.

# Parallelization strategy

Since it's a sum of $N$ elements, it can be splitted into $m$ different threads, each with $m / N$ sums to compute.

<div style="display: flex; align-items: center; width: 100%;">
  <figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="./images/omp.png" alt="OMP" width="80%" />
  </figure>
</div>

# Implementation details in SW2 👨🏻‍💻
lorem ipsum

# Hotspot analysis 🔥

The only hotspot that is worth mentioning is the `loop in DFT at omp_homework.c:71`, as it takes $\approx 98\%$ of the computation time. The said loop is the following:

```c
for (k = 0; k < N; k++) {
   for (n = 0; n < N; n++) {
   // Real part of X[k]
   Xr_o[k] += xr[n] * cos(n * k * PI2 / N) + idft * xi[n] * sin(n * k * PI2 / N);
   // Imaginary part of X[k]
   Xi_o[k] += -idft * xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
   }
}
```

Since it's in $O(N^2)$, massaging this section is crucial to speed up the program.

OLD {
inside the vanilla version of the code, the hotspot is on the line 70 (the nested outer for loop takes 13.2 seconds out of the 13.44 seconds of total execution time)
inside the vanilla version of the code, the hotspot is on the line 71 (the nested inner for loop takes 13.2 seconds out of the 13.44 seconds of total execution time)
it is a scalar loop that can be parallelized with OpenMP
}

# Compiler optimization 🎭

OLD{
explain all the flags used in the makefile
}

We compiled the program using the following command:

```bash
icx -g -Wall -std=c99 -qopenmp -qopt-report=3 -xHost -O3 -ffast-math omp_homework.c
```

In this way, the code is properly optimized, the best istruction set is used and the program is vectorized when possible.

# caching?? 📒
lorem ipsum

# Vectorization 🏹
using vectorization as in the code inside omp homework, the execution time is 0.71 seconds, which is 18.5 times faster than the vanilla version of the code (for N=10000) with a non existent error (Xre = 10000.0000)


# Parallelization 🛤️

OLD{
using 
   ifndef PARALLEL
      omp set num threads(4)
      pragma omp parallel for private (k)
   endif

using "omp parallel (k,n)" k and n are private by default 
}

The first thing we changed was the `sin` and `cos` computation. Those are used in both the statements inside the inner loop, so they can be extracted to variables

```c
double cos_res = cos(n * k * PI2 / N);
double sin_res = sin(n * k * PI2 / N);

// Real part of X[k]
Xr_o[k] += xr[n] * cos_res + idft * xi[n] * sin_res;
// Imaginary part of X[k]
Xi_o[k] += -idft * xr[n] * sin_res + xi[n] * cos_res;
```

Then, there's the actual parallelization part. We used OpenMP as follows:

```c
#pragma omp parallel for num_threads(NTHREADS) collapse(2) schedule(static) reduction(+ : Xr_o[ : N], Xi_o[ : N])
for (k = 0; k < N; k++) {
   for (n = 0; n < N; n++) {
   double cos_res = cos(n * k * PI2 / N);
   double sin_res = sin(n * k * PI2 / N);

   // Real part of X[k]
   Xr_o[k] += xr[n] * cos_res + idft * xi[n] * sin_res;
   // Imaginary part of X[k]
   Xi_o[k] += -idft * xr[n] * sin_res + xi[n] * cos_res;
   }
}
```

In order, we collapsed the nested loops in a single $N \times N$ loop managed by OMP, then with the `schedule(static)` we stated that every thread would have had the same chunk size to work on, and finally the `reduction` clause aggregates the sums on the arrays.


# Performance evaluation 🤔
vectorization + Parallelization takes to around 70 seconds of execution time vs the original 17 seconds

# Conclusions 🔚
lorem ipsum