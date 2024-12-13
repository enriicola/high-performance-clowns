# HPC - OpenMP report 🏎️ 💻
### Leonardo Gonfiantini,  Christian Parodi, Enrico Pezzano
### December 2024

# Introduction 🎬
The goal of this laboratory is to optimize the implementation of the Discrete Fourier Transform (DFT) algorithm by leveraging OpenMP HPC techniques, with a focus on parallelization and vectorization, especially the hotspots.

bozza{
    Specifically, the optimization process employs OpenMP to improve performance and scalability. The primary objectives include reducing execution time, maintaining computational accuracy, and addressing challenges such as hotspot identification, vectorization issues, and thread scalability. Through this approach, various strategies—including hotspot analysis, vectorization, and parallelization—are explored and evaluated to enhance the efficiency of the DFT implementation.
}

# Implementation details in SW2 👨🏻‍💻
lorem ipsum

# Hotspot analysis 🔥
inside the vanilla version of the code, the hotspot is on the line 70 (the nested outer for loop takes 13.2 seconds out of the 13.44 seconds of total execution time)
inside the vanilla version of the code, the hotspot is on the line 71 (the nested inner for loop takes 13.2 seconds out of the 13.44 seconds of total execution time)
it is a scalar loop that can be parallelized with OpenMP


# Performance optimization 🎭
explain all the flags used in the makefile


# caching?? 📒
lorem ipsum

# Vectorization 🏹
using vectorization as in the code inside omp homework, the execution time is 0.71 seconds, which is 18.5 times faster than the vanilla version of the code (for N=10000) with a non existent error (Xre = 10000.0000)


# Parallelization 🛤️
using 
   ifndef PARALLEL
      omp set num threads(4)
      pragma omp parallel for private (k)
   endif

using "omp parallel (k,n)" k and n are private by default 


# Performance evaluation 🤔
vectorization + Parallelization takes to around 70 seconds of execution time vs the original 17 seconds

# Conclusions 🔚
lorem ipsum