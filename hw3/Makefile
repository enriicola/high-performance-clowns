NPROCS=10
RELEASE_PATH=release
SOURCE_PATH=source
FLAGS=-g -O2 -xHost -qopenmp -qopt-report=3 -ffast-math

# default = best sequential time
default:
	mpiicx ${FLAGS} ./${SOURCE_PATH}/pi_homework.c -o ./${RELEASE_PATH}/pi_homework

# normal sequential time
# no-vec:
# 	mpiicx -g ./${SOURCE_PATH}/pi_homework.c -o ./${RELEASE_PATH}/pi_homework

parallel:
	mpiicx -D PARALLEL ${FLAGS} ./${SOURCE_PATH}/pi_homework.c -o ./${RELEASE_PATH}/pi_homework

mpi:
	mpiicx ${FLAGS} ./${SOURCE_PATH}/pi_homework_mpi.c -o ./${RELEASE_PATH}/pi_homework_mpi
	mpirun -np ${NPROCS} ./${RELEASE_PATH}/pi_homework_mpi
