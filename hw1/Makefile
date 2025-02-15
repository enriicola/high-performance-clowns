CC=icx
FLAGS=-g -Wall -qopenmp -qopt-report=3
SOURCE=./src
RELEASE=./release

SIZE ?= 20000
# OMP
NTHREADS ?= 20
OMP_SCHEDULE ?= dynamic

default:
	${CC} ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

# best sequential
vec: FLAGS += -xHost -O3 -ffast-math
vec: 
	${CC} ${FLAGS} -D SIZE=${SIZE} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

parallel: FLAGS += -xHost -O3 -ffast-math
parallel: 
	${CC} ${FLAGS} \
		-D SIZE=${SIZE} \
		-D PARALLEL \
		-D OMP_SCHEDULE=${OMP_SCHEDULE} \
		-D NTHREADS=${NTHREADS} \
		${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework 

clean:
	rm -rf ${RELEASE}/*