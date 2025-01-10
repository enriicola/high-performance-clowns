CC=nvcc
SOURCE=./src
RELEASE=./release
BLOCKDIM_X ?= 16
BLOCKDIM_Y ?= 16
FLAGS=-O3 -x cu
ARCH=sm_86 # CUDA Capability Major/Minor version number: 8.6
SIZE ?= 1000 # row = column = size
NSTEPS ?= 200

default:
	${CC} ${FLAGS} -arch=${ARCH} \
		-D BLOCKDIM_X=${BLOCKDIM_X} \
		-D BLOCKDIM_Y=${BLOCKDIM_Y} \
		-D SIZE=${SIZE} \
		-D NSTEPS=${NSTEPS} \
		${SOURCE}/heat.cu -o ${RELEASE}/heat_cuda -run

# -march=native == -xHost
seq:
	gcc ${FLAGS} -march=native \
		-D SIZE=${SIZE} \
		-D NSTEPS=${NSTEPS} \
		${SOURCE}/heat.c -o ${RELEASE}/heat && ${RELEASE}/heat