CC=nvcc
SOURCE=./src
RELEASE=./release

default:
	${CC} ${SOURCE}/heat.cu -o ${RELEASE}/heat_cuda -run

seq:
	gcc  -Wall ${SOURCE}/heat.c -o ${RELEASE}/heat