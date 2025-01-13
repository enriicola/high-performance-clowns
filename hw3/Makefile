NPROCS ?= 10
RELEASE_PATH=release
SOURCE_PATH=src
FLAGS=-g -O3 -xHost -qopenmp -qopt-report=3 -ffast-math
INTERVALS ?= 100000000000

# default = best sequential time
default:
	mpiicx ${FLAGS} \
	-D PARALLEL \
	-D INTERVALS=${INTERVALS} \
	./${SOURCE_PATH}/pi_homework.c -o ./${RELEASE_PATH}/pi_homework

mpi:
	mpiicx ${FLAGS} \
	-D INTERVALS=${INTERVALS} \
	./${SOURCE_PATH}/pi_homework_mpi.c -o ./${RELEASE_PATH}/pi_homework_mpi
	mpirun -np ${NPROCS} ./${RELEASE_PATH}/pi_homework_mpi
