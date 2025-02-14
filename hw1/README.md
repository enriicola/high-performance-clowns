The goal of this homework is to parallelise/vectorise the following program corresponding to an implementation of the Discrete Fourier Transform algorithm.

**The focus is on writing an exhaustive report about the code analysis (i.e. hotspot identification) and on the iterative process to speedup the application.**

You should discuss at least the following points:

1. hotspot identification with NUMBERS. Don't say only that the main hotspot is the for loop in lines xx-yy, but that is a hotspot because it requires e.g. 80 seconds for a global time of 100 seconds;
2. discuss possible vectorization issues with the report provided by the Intel compiler;
3. define the BEST sequential time to be used as reference;
4. present the performance using a PROPER number of threads and data size on the lab workstation (i.e. SW2), NOT on your PC;
    1. provide tables and charts regarding speedup and efficiency;
    2. do not consider data size resulting in sequential execution time < 30 seconds;
    3. discuss conclusions. 

The use of your pc is allowed ONLY if you plan to explore ROCm on a AMD cpu. Otherwise you can develop on your PC but the tests have to be performed in SW1 or SW2 AND using the Intel compiler.

Upload a zip file containing the source code and the report in pdf and send the same via email.