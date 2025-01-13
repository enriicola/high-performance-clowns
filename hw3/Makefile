NPROCS ?= 20
RELEASE=./release
SOURCE=./src
FLAGS=-g -O3 -xHost
INTERVALS ?= 100000000000

default: FLAGS +=-qopenmp -qopt-report=3 -ffast-math
default:
	mpiicx ${FLAGS} \
	-D PARALLEL \
	-D INTERVALS=${INTERVALS} \
	${SOURCE}/pi_homework.c -o ${RELEASE}/pi_homework

mpi:
	mpiicx ${FLAGS} \
	-D INTERVALS=${INTERVALS} \
	${SOURCE}/pi_homework_mpi.c -o ${RELEASE}/pi_homework_mpi
	mpirun -np ${NPROCS} ${RELEASE}/pi_homework_mpi
