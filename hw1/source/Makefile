CC = icx
CFLAGS = -c -g -qopenmp
LDFLAGS = -qopenmp
OBJ = main.o
TARGET = ./../release/main

default: release

release: $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o $(TARGET)

$(OBJ): omp_homework.c
	$(CC) $(CFLAGS) omp_homework.c -o $(OBJ)

parallel: CFLAGS += -O2 -xHost
parallel: $(OBJ)
	$(CC) -D PARALLEL $(LDFLAGS) $(OBJ) -o $(TARGET)

vector: CFLAGS += -O2 -xHost
vector: $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)

run: release
	$(TARGET)