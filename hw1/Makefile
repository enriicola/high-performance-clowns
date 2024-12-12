CC=icx
FLAGS=-g -qopenmp 
SOURCE=./source
RELEASE=./release

default:
	${CC} ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

# best sequential
vec: FLAGS += -xHost -O2 -ffast-math
vec: 
	${CC} ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

parallel: FLAGS += -xHost -O2 -ffast-math
parallel: 
	${CC} -D PARALLEL ${FLAGS} ${SOURCE}/omp_homework.c -o ${RELEASE}/omp_homework

clean:
	rm -rf ${RELEASE}/*