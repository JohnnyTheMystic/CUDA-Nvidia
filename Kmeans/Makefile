IDIR   = ./
ODIR   = ./Obj/
SDIR   = ./
LIBS   = -lm -fopenmp -O3
INCLUDES = -I/usr/local/cuda-9.2/include
LDNVCC = -L/usr/local/cuda-9.2/lib64
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_SM52    := -gencode arch=compute_52,code=sm_52
GENCODE_FLAGS   := $(GENCODE_SM35) 


all: build

build: run

main.o: main.c
	g++ $(LIBS) -c -o $(ODIR)main.o main.c

kmeans.o: kmeans.c
	g++ $(LIBS) -c -o $(ODIR)kmeans.o kmeans.c

File.o: File.c
	g++ $(LIBS) -c -o $(ODIR)File.o File.c

kmeansgpu.o: kmeansgpu.cu
	nvcc -O3 -m64 $(INCLUDES) $(GENCODE_FLAGS) -o $(ODIR)kmeansgpu.o -c -w kmeansgpu.cu 

run: main.o File.o kmeans.o kmeansgpu.o
	g++ $(LIBS) $(LDNVCC) -o run $(ODIR)main.o $(ODIR)File.o $(ODIR)kmeans.o $(ODIR)kmeansgpu.o -lcudart

clean:
	rm -f $(ODIR)*.o *~ core $(INCDIR)/*~ run $(SDIR)*~
