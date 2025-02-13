CC=icpx
FLAGS=-g -O3 -march=native -qopenmp -qopt-report=3 -ffast-math
SOURCE=./src
RELEASE=./release
ARCH ?= sm_86 # check with devicequery
RESOLUTION ?= 1000
# OMP
NUM_THREADS ?= 4
OMP_SCHEDULE ?= dynamic
# CUDA
BLOCKDIM_X ?= 32
BLOCKDIM_Y ?= 32

default:
	$(CC) $(FLAGS) \
		-D PARALLEL \
		-D NUM_THREADS=$(NUM_THREADS) \
		-D OMP_SCHEDULE=$(OMP_SCHEDULE) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot

seq:
	$(CC) $(FLAGS) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot

cuda:
	nvcc -O3 -arch=$(ARCH) \
		-D BLOCKDIM_X=$(BLOCKDIM_X) \
		-D BLOCKDIM_Y=$(BLOCKDIM_Y) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot.cu -o $(RELEASE)/mandelbrot_cuda