CC=icpx
FLAGS=-g -O3 -xHost -qopenmp -qopt-report=3 -ffast-math
SOURCE=./src
RELEASE=./release

NUM_THREADS ?= 4
RESOLUTION ?= 1000
SHOW ?= 0 # $(if $(filter 0,$(SHOW)),, -D SHOW) -> if SHOW != 0 display the result

default:
	$(CC) $(FLAGS) \
		-D PARALLEL \
		-D NUM_THREADS=$(NUM_THREADS) \
		-D RESOLUTION=$(RESOLUTION) \
		$(if $(filter 0,$(SHOW)),,-D SHOW) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot

seq:
	$(CC) $(FLAGS) \
		-D NUM_THREADS=$(NUM_THREADS) \
		-D RESOLUTION=$(RESOLUTION) \
		$(if $(filter 0,$(SHOW)),,-D SHOW) \
		$(SOURCE)/mandelbrot.cpp -o $(RELEASE)/mandelbrot