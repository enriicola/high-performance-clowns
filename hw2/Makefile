CC=nvcc
SOURCE=./src
RELEASE=./release
BLOCKDIM_X ?= 32
BLOCKDIM_Y ?= 8
FLAGS=-O3
ARCH ?=sm_86 # CUDA Capability Major/Minor version number: 8.6
SIZE ?= 1000 # row = column = size
NTRHEADS ?= 2

default:
	${CC} ${FLAGS} -arch=${ARCH} \
		-D BLOCKDIM_X=${BLOCKDIM_X} \
		-D BLOCKDIM_Y=${BLOCKDIM_Y} \
		-D SIZE=${SIZE} \
		${SOURCE}/heat.cu -o ${RELEASE}/heat_cuda -run

cpu: FLAGS += -march=native -fopenmp
cpu:
	gcc ${FLAGS} \
    -D PARALLEL \
    -D NTRHEADS=${NTRHEADS} \
		-D SIZE=${SIZE} \
		${SOURCE}/heat.c -o ${RELEASE}/heat