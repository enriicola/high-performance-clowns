
# prof instruction
--------------------------------------------------------
Write a program that accelerates the attached program that computes the 2D heat conduction formula (see jpg).

Evaluate different configurations <<<block, thread>>> AND different problem sizes.

As for the OMP homework, the focus is on writing an exhaustive report about the code analysis (i.e. hotspot identification) and the iterative process to speed up the application.

You should discuss at least the following points:

- hotspot identification with NUMBERS. Don't say only that the main hotspot is the for loop in lines xx-yy, but that is a hotspot because it requires e.g. 80 seconds for a global time of 100 seconds;
- discuss possible vectorization issues with the report provided by the Intel compiler;
- define the BEST sequential time to be used as reference;
- present the performance using Google colab;
- provide tables and charts regarding speedup values only;
- do not consider data size resulting in sequential execution time < 30 seconds;
- discuss conclusions.
The use of your pc is allowed ONLY if you plan to explore ROCm on a AMD cpu. Otherwise you can develop on your PC but the tests have to be performed with Colab, the sequential ones using gcc .

Upload a zip file containing the notebook, other possible files and the report in pdf. Send the same via email.

The compute capabilities of the GPU in Colab can be retrieved using the devicequery program.
--------------------------------------------------------

# Introduction
In this report, we aim to accelerate a given program that computes the 2D heat conduction formula using CUDA. The primary objective is to evaluate different configurations of blocks and threads, as well as various problem sizes, to achieve optimal performance. We will conduct a thorough analysis of the code to identify hotspots and discuss potential vectorization issues. The best sequential time will be established as a reference point, and performance metrics will be presented using Google Colab. The report will include tables and charts to illustrate speedup values, focusing on data sizes that result in sequential execution times of 30 seconds or more. Finally, we will draw conclusions based on our findings.


# Hardware Capability ⚙️
For this first assignment, we executed the C code using the Software 2 (SW2) workstations, with the following characteristics.

# Hotspot identification



# Vectorization issues

```

```

# Best sequential time

# Google colab graphs

# Speed up charts and tables
- do not consider data size resulting in sequential execution time < 30 seconds;

# Conclusions
