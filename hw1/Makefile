CC=icx
FLAGS=-g -Wall -qopenmp -qopt-report=3
SOURCE=./src
RELEASE=./release
NTHREADS ?= 20

default:
	${CC} ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

# best sequential
vec: FLAGS += -xHost -O3 -ffast-math
vec: 
	${CC} ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

parallel: FLAGS += -xHost -O3 -ffast-math
parallel: 
	${CC} -D PARALLEL -D NTHREADS=${NTHREADS} ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework 

clean:
	rm -rf ${RELEASE}/*