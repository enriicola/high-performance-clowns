CC=icpx
FLAGS=-g -O3 -qopenmp -qopt-report=3 -ffast-math
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

default: FLAGS += -march=native
default:
	$(CC) $(FLAGS) \
		-D PARALLEL \
		-D NUM_THREADS=$(NUM_THREADS) \
		-D OMP_SCHEDULE=$(OMP_SCHEDULE) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot

avx2: FLAGS += -xCORE-AVX2
avx2:
	$(CC) $(FLAGS) \
		-D PARALLEL \
		-D NUM_THREADS=$(NUM_THREADS) \
		-D OMP_SCHEDULE=$(OMP_SCHEDULE) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot_AVX2.cpp -o $(RELEASE)/mandelbrot_AVX2

avx2-seq: FLAGS += -xCORE-AVX2
avx2-seq:
	$(CC) $(FLAGS) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot_AVX2.cpp -o $(RELEASE)/mandelbrot_AVX2_seq

seq: FLAGS += -march=native
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