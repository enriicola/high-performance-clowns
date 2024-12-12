The goal of this homework is to parallelise/vectorise the following program that computes the pi value. 

The focus is on writing an exhaustive report about the code analysis (i.e. hotspot identification) and on the iterative process to speedup the application.

You should discuss at least the following points:

hotspot identification with NUMBERS. Don't say only that the main hotspot is the for loop in lines xx-yy, but that is a hotspot because it requires e.g. 80 seconds for a global time of 100 seconds;
discuss possible vectorization issues with the report provided by the Intel compiler;
define the BEST sequential time to be used as reference;
present the performance using a PROPER number of PROCESSES and data size on the lab workstation (i.e. SW2), NOT on your PC;
provide tables and charts regarding speedup and efficiency;
do not consider data size resulting in sequential execution time < 30 seconds;
discuss conclusions. 
This year we don't have a cluster available. Therefore use one workstation in SW1 or SW2 AND using the Intel compiler.

Remember
source /opt/intel/oneapi/setvars.sh

Upload a zip file containing the source code and the report in pdf and send the same via email.

# Reports not compliant with the above indications will be rejected.