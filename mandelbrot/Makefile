CC=icpx
FLAGS=-g -O3 -xHost -qopenmp -qopt-report=3 -ffast-math
SOURCE=./src
RELEASE=./release
ARCH ?= sm_86 # check with devicequery
RESOLUTION ?= 1000

default:
	$(CC) $(FLAGS) \
		-D PARALLEL \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot

seq:
	$(CC) $(FLAGS) \
		-D RESOLUTION=$(RESOLUTION) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot

cuda:
	nvcc -O3 -arch=$(ARCH) \
		-D RESOLUTION=${RESOLUTION} \
		$(SOURCE)/mandelbrot.cu -o $(RELEASE)/mandelbrot_cuda